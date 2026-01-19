/**
 * @file json.h
 * @author Neil (github: nbsdx)
 * @brief Obtained from https://github.com/nbsdx/SimpleJSON , under the terms of
 * the WTFPL.
 * @date 2024-02-22
 */

#include "json.h"

namespace json {

namespace {
string json_escape(const string &str) {
  string output;
  for (unsigned i = 0; i < str.length(); ++i) switch (str[i]) {
      case '\"':
        output += "\\\"";
        break;
      case '\\':
        output += "\\\\";
        break;
      case '\b':
        output += "\\b";
        break;
      case '\f':
        output += "\\f";
        break;
      case '\n':
        output += "\\n";
        break;
      case '\r':
        output += "\\r";
        break;
      case '\t':
        output += "\\t";
        break;
      default:
        output += str[i];
        break;
    }
  return output;
}

JSON parse_next(const string &, size_t &);

void consume_ws(const string &str, size_t &offset) {
  while (isspace(str[offset])) ++offset;
}

JSON parse_object(const string &str, size_t &offset) {
  JSON Object = JSON::Make(JSON::Class::Object);

  ++offset;
  consume_ws(str, offset);
  if (str[offset] == '}') {
    ++offset;
    return Object;
  }

  while (true) {
    JSON Key = parse_next(str, offset);
    consume_ws(str, offset);
    if (str[offset] != ':') {
      // std::cerr << "Error: Object: Expected colon, found '" << str[offset] <<
      // "'\n";
      break;
    }
    consume_ws(str, ++offset);
    JSON Value = parse_next(str, offset);
    Object[Key.ToString()] = Value;

    consume_ws(str, offset);
    if (str[offset] == ',') {
      ++offset;
      continue;
    } else if (str[offset] == '}') {
      ++offset;
      break;
    } else {
      // std::cerr << "ERROR: Object: Expected comma, found '" << str[offset] <<
      // "'\n";
      break;
    }
  }

  return Object;
}

JSON parse_array(const string &str, size_t &offset) {
  JSON Array = JSON::Make(JSON::Class::Array);
  unsigned index = 0;

  ++offset;
  consume_ws(str, offset);
  if (str[offset] == ']') {
    ++offset;
    return Array;
  }

  while (true) {
    Array[index++] = parse_next(str, offset);
    consume_ws(str, offset);

    if (str[offset] == ',') {
      ++offset;
      continue;
    } else if (str[offset] == ']') {
      ++offset;
      break;
    } else {
      // std::cerr << "ERROR: Array: Expected ',' or ']', found '" <<
      // str[offset] << "'\n";
      return std::move(JSON::Make(JSON::Class::Array));
    }
  }

  return Array;
}

JSON parse_string(const string &str, size_t &offset) {
  JSON String;
  string val;
  for (char c = str[++offset]; c != '\"'; c = str[++offset]) {
    if (c == '\\') {
      switch (str[++offset]) {
        case '\"':
          val += '\"';
          break;
        case '\\':
          val += '\\';
          break;
        // case '/' : val += '/' ; break;
        case 'b':
          val += '\b';
          break;
        case 'f':
          val += '\f';
          break;
        case 'n':
          val += '\n';
          break;
        case 'r':
          val += '\r';
          break;
        case 't':
          val += '\t';
          break;
        case 'u': {
          val += "\\u";
          for (unsigned i = 1; i <= 4; ++i) {
            c = str[offset + i];
            if ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
                (c >= 'A' && c <= 'F'))
              val += c;
            else {
              // std::cerr << "ERROR: String: Expected hex character in unicode
              // escape, found '" << c << "'\n";
              return std::move(JSON::Make(JSON::Class::String));
            }
          }
          offset += 4;
        } break;
        default:
          val += '\\';
          break;
      }
    } else
      val += c;
  }
  ++offset;
  String = val;
  return String;
}

JSON parse_number(const string &str, size_t &offset) {
  JSON Number;
  string val, exp_str;
  char c;
  bool isDouble = false;
  long exp = 0;
  while (true) {
    c = str[offset++];
    if ((c == '-') || (c >= '0' && c <= '9'))
      val += c;
    else if (c == '.') {
      val += c;
      isDouble = true;
    } else
      break;
  }
  if (c == 'E' || c == 'e') {
    c = str[offset++];
    if (c == '-') {
      ++offset;
      exp_str += '-';
    }
    while (true) {
      c = str[offset++];
      if (c >= '0' && c <= '9')
        exp_str += c;
      else if (!isspace(c) && c != ',' && c != ']' && c != '}') {
        // std::cerr << "ERROR: Number: Expected a number for exponent, found '"
        // << c << "'\n";
        return std::move(JSON::Make(JSON::Class::Null));
      } else
        break;
    }
    exp = std::stol(exp_str);
  } else if (!isspace(c) && c != ',' && c != ']' && c != '}') {
    // std::cerr << "ERROR: Number: unexpected character '" << c << "'\n";
    return std::move(JSON::Make(JSON::Class::Null));
  }
  --offset;

  if (isDouble)
    Number = std::stod(val) * std::pow(10, exp);
  else {
    if (!exp_str.empty())
      Number = (double)std::stol(val) * std::pow(10, exp);
    else
      Number = std::stol(val);
  }
  return Number;
}

JSON parse_bool(const string &str, size_t &offset) {
  JSON Bool;
  if (str.substr(offset, 4) == "true")
    Bool = true;
  else if (str.substr(offset, 5) == "false")
    Bool = false;
  else {
    // std::cerr << "ERROR: Bool: Expected 'true' or 'false', found '" <<
    // str.substr( offset, 5 ) << "'\n";
    return std::move(JSON::Make(JSON::Class::Null));
  }
  offset += (Bool.ToBool() ? 4 : 5);
  return Bool;
}

JSON parse_null(const string &str, size_t &offset) {
  JSON Null;
  if (str.substr(offset, 4) != "null") {
    // std::cerr << "ERROR: Null: Expected 'null', found '" << str.substr(
    // offset, 4 ) << "'\n";
    return std::move(JSON::Make(JSON::Class::Null));
  }
  offset += 4;
  return Null;
}

JSON parse_next(const string &str, size_t &offset) {
  char value;
  consume_ws(str, offset);
  value = str[offset];
  switch (value) {
    case '[':
      return std::move(parse_array(str, offset));
    case '{':
      return std::move(parse_object(str, offset));
    case '\"':
      return std::move(parse_string(str, offset));
    case 't':
    case 'f':
      return std::move(parse_bool(str, offset));
    case 'n':
      return std::move(parse_null(str, offset));
    default:
      if ((value <= '9' && value >= '0') || value == '-')
        return std::move(parse_number(str, offset));
  }
  // std::cerr << "ERROR: Parse: Unknown starting character '" << value <<
  // "'\n"; printf("ERROR: Parse: Unknown starting character %02hhX\n", value);
  return JSON();
}
}  // anonymous namespace

string JSON::ToString(bool &ok) const {
  ok = (Type == Class::String);
  // return ok ? std::move( json_escape( *Internal.String ) ): string("");
  // return ok ? *Internal.String: string("");
  return ok ? *Internal.String : dump();
}

string JSON::dump(int depth, string tab) const {
  string pad = "";
  for (int i = 0; i < depth; ++i, pad += tab);

  switch (Type) {
    case Class::Null:
      return "null";
    case Class::Object: {
      string s = "{\n";
      bool skip = true;
      for (auto &p : *Internal.Map) {
        if (!skip) s += ",\n";
        s += (pad + "\"" + p.first + "\" : " + p.second.dump(depth + 1, tab));
        skip = false;
      }
      s += ("\n" + pad.erase(0, 2) + "}");
      return s;
    }
    case Class::Array: {
      string s = "[";
      bool skip = true;
      for (auto &p : *Internal.List) {
        if (!skip) s += ", ";
        s += p.dump(depth + 1, tab);
        skip = false;
      }
      s += "]";
      return s;
    }
    case Class::String:
      return "\"" + json_escape(*Internal.String) + "\"";
    case Class::Floating:
      return std::to_string(Internal.Float);
    case Class::Integral:
      return std::to_string(Internal.Int);
    case Class::Boolean:
      return Internal.Bool ? "true" : "false";
    default:
      return "";
  }
  return "";
}
JSON JSON::Load(const string &str) {
  size_t offset = 0;
  return std::move(parse_next(str, offset));
}

JSON Array() { return std::move(JSON::Make(JSON::Class::Array)); }

template <typename... T>
JSON Array(T... args) {
  JSON arr = JSON::Make(JSON::Class::Array);
  arr.append(args...);
  return std::move(arr);
}

JSON Object() { return std::move(JSON::Make(JSON::Class::Object)); }

}  // namespace json
