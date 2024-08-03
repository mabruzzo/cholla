#include "../io/ParameterMap.h"

#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>

#include "../global/global.h"  // MAXLEN
#include "../io/io.h"          // chprintf
#include "../utils/error_handling.h"

[[noreturn]] void param_details::Report_TypeErr_(const std::string& param, const std::string& str,
                                                 const std::string& dtype, param_details::TypeErr type_convert_err)
{
  std::string r;
  using param_details::TypeErr;
  switch (type_convert_err) {
    case TypeErr::none:
      r = "";
      break;  // this shouldn't happen
    case TypeErr::generic:
      r = "invalid value";
      break;
    case TypeErr::boolean:
      r = R"(boolean values must be "true" or "false")";
      break;
    case TypeErr::out_of_range:
      r = "out of range";
      break;
  }
  CHOLLA_ERROR("error interpretting \"%s\", the value of the \"%s\" parameter, as a %s: %s", str.c_str(), param.c_str(),
               dtype.c_str(), r.c_str());
}

param_details::TypeErr param_details::try_bool_(const std::string& str, bool& val)
{
  if (str == "true") {
    val = true;
  } else if (str == "false") {
    val = false;
  } else {
    return param_details::TypeErr::boolean;
  }
  return param_details::TypeErr::none;
}

param_details::TypeErr param_details::try_int64_(const std::string& str, std::int64_t& val)
{
  char* ptr_end{};
  errno         = 0;  // reset errno to 0 (prior library calls could have set it to an arbitrary value)
  long long tmp = std::strtoll(str.data(), &ptr_end, 10);  // the last arg specifies base-10

  if (errno == ERANGE) {  // deal with errno first, so we don't accidentally overwrite it
    // - non-zero vals other than ERANGE are implementation-defined (plus, the info is redundant)
    return param_details::TypeErr::out_of_range;
  } else if ((str.data() + str.size()) != ptr_end) {
    // when str.data() == ptr_end, then no conversion was performed.
    // when (str.data() + str.size()) != ptr_end, str could hold a float or look like "123abc"
    return param_details::TypeErr::generic;
#if (LLONG_MIN != INT64_MIN) || (LLONG_MAX != INT64_MAX)
  } else if ((tmp < INT64_MIN) and (tmp > INT64_MAX)) {
    return param_details::TypeErr::out_of_range;
#endif
  }
  val = std::int64_t(tmp);
  return param_details::TypeErr::none;
}

param_details::TypeErr param_details::try_double_(const std::string& str, double& val)
{
  char* ptr_end{};
  errno = 0;  // reset errno to 0 (prior library calls could have set it to an arbitrary value)
  val   = std::strtod(str.data(), &ptr_end);

  if (errno == ERANGE) {  // deal with errno first, so we don't accidentally overwrite it
    // - non-zero vals other than ERANGE are implementation-defined (plus, the info is redundant)
    return param_details::TypeErr::out_of_range;
  } else if ((str.data() + str.size()) != ptr_end) {
    // when str.data() == ptr_end, then no conversion was performed.
    // when (str.data() + str.size()) != ptr_end, str could look like "123abc"
    return param_details::TypeErr::generic;
  }
  return param_details::TypeErr::none;
}

param_details::TypeErr param_details::try_string_(const std::string& str, std::string& val)
{
  // mostly just exists for consistency (every parameter can be considered a string)
  // note: we may want to consider removing surrounding quotation marks in the future
  val = str;  // we make a copy for the sake of consistency
  return param_details::TypeErr::none;
}

namespace
{  // stuff inside an anonymous namespace is local to this file

/*! Helper class that specifes the parts of a string correspond to the key and the value */
struct KeyValueViews {
  std::string_view key;
  std::string_view value;
};

/*! \brief Try to extract the parts of nul-terminated c-string that refers to a parameter name
 *  and a parameter value. If there are any issues, views will be empty optional is returned. */
KeyValueViews Try_Extract_Key_Value_View(const char* buffer)
{
  // create a view that wraps the full buffer (there aren't any allocations)
  std::string_view full_view(buffer);

  // we explicitly mimic the old behavior

  // find the position of the equal sign
  std::size_t pos = full_view.find('=');

  // handle the edge-cases (where we can't parse a key-value pair)
  if ((pos == 0) or                       // '=' sign is the first character
      ((pos + 1) == full_view.size()) or  // '=' sign is the last character
      (pos == std::string_view::npos)) {  // there is no '=' sign
    return {std::string_view(), std::string_view()};
  }
  return {full_view.substr(0, pos), full_view.substr(pos + 1)};
}

/*! \brief Modifies the string_view to remove trailing and leading whitespace.
 *
 *  \note
 *  Since this is a string_view, we don't actually mutate any characters
 */
void my_trim(std::string_view& s)
{
  /* Trim left side */
  std::size_t start           = 0;
  const std::size_t max_start = s.size();
  while ((start < max_start) and std::isspace(s[start])) {
    start++;
  }
  if (start > 0) s = s.substr(start);

  /* Trim right side */
  std::size_t cur_len = s.size();
  while ((cur_len > 0) and std::isspace(s[cur_len - 1])) {
    cur_len--;
  }
  if (cur_len < s.size()) s = s.substr(0, cur_len);
}

}  // anonymous namespace

ParameterMap::ParameterMap(std::FILE* fp, int argc, char** argv)
{
  int buf;
  char *s, buff[256];

  CHOLLA_ASSERT(fp != nullptr, "ParameterMap was passed a nullptr rather than an actual file object");

  /* Read next line */
  while ((s = fgets(buff, sizeof buff, fp)) != NULL) {
    /* Skip blank lines and comments */
    if (buff[0] == '\n' || buff[0] == '#' || buff[0] == ';') {
      continue;
    }

    /* Parse name/value pair from line */
    KeyValueViews kv_pair = Try_Extract_Key_Value_View(buff);
    // skip this line if there were any parsing errors (I think we probably abort with an
    // error instead, but we are currently maintaining historical behavior)
    if (kv_pair.key.empty()) continue;
    my_trim(kv_pair.value);
    entries_[std::string(kv_pair.key)] = {std::string(kv_pair.value), false};
  }

  // Parse overriding args from command line
  for (int i = 0; i < argc; ++i) {
    // try to parse the argument
    KeyValueViews kv_pair = Try_Extract_Key_Value_View(argv[i]);
    if (kv_pair.key.empty()) continue;
    my_trim(kv_pair.value);
    // coerce to string first so we can print
    std::string key_str(kv_pair.key);
    std::string value_str(kv_pair.value);
    chprintf("Override with %s=%s\n", key_str.c_str(), value_str.c_str());
    entries_[key_str] = {value_str, false};
  }
}

int ParameterMap::warn_unused_parameters(const std::set<std::string>& ignore_params, bool abort_on_warning,
                                         bool suppress_warning_msg) const
{
  int unused_params = 0;
  for (const auto& kv_pair : entries_) {
    const std::string& name                     = kv_pair.first;
    const ParameterMap::ParamEntry& param_entry = kv_pair.second;

    if ((not param_entry.accessed) and (ignore_params.find(name) == ignore_params.end())) {
      unused_params++;
      const std::string& value = param_entry.param_str;
      if (abort_on_warning) {
        CHOLLA_ERROR("%s/%s:  Unknown parameter/value pair!", name.c_str(), value.c_str());
      } else if (not suppress_warning_msg) {
        chprintf("WARNING: %s/%s:  Unknown parameter/value pair!\n", name.c_str(), value.c_str());
      }
    }
  }
  return unused_params;
}