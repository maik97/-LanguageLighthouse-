import json


class UnexpectedJSONFormatError(Exception):
    pass


def validate_json_format(response, json_template):

    def compare_structures(obj, template, location):

        if type(obj) != type(template):
            raise UnexpectedJSONFormatError(f"Expected type {type(template)} at {location}, got: {type(obj)}")

        if isinstance(template, dict):
            if '...' in template:
                if not obj:
                    raise UnexpectedJSONFormatError(f"Expected a non-empty dictionary at {location}, got an empty dictionary")
                template_key = list(template.keys())[0]
                template_value = template[template_key]
                for key, value in obj.items():
                    compare_structures(value, template_value, location + f".{key}")
            else:
                if len(template) != len(obj):
                    raise UnexpectedJSONFormatError(f"Expected dictionary with {len(template)} keys at {location}, got {len(obj)}")
                for key, value in template.items():
                    if key not in obj:
                        raise UnexpectedJSONFormatError(f"Missing key '{key}' at {location} in JSON object: {obj}")
                    compare_structures(obj[key], value, location + f".{key}")

        elif isinstance(template, list):
            if '...' in template:
                if not obj:
                    raise UnexpectedJSONFormatError(f"Expected a non-empty list at {location}, got an empty list")
                template_value = template[0]
                for index, item in enumerate(obj):
                    compare_structures(item, template_value, location + f"[{index}]")
            else:
                if len(obj) != len(template):
                    raise UnexpectedJSONFormatError(f"Expected list of length {len(template)} at {location}, got: {len(obj)}")
                for index, (item, template_item) in enumerate(zip(obj, template)):
                    compare_structures(item, template_item, location + f"[{index}]")

    json_obj = json.loads(response)
    template_obj = json.loads(json_template)
    compare_structures(json_obj, template_obj, "root")
    return json_obj