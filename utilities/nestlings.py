from collections import UserDict
from collections.abc import Mapping, Iterable
from functools import reduce


class Nest(UserDict):
    def __init__(self, data=None, include=None, exclude=None, stop_depth=None, stop_keys=None):
        self.parent_nest = None
        if not data:
            super().__init__()
        else:
            flattened_data, self._blueprint = flatten(
                data,
                include=include,
                exclude=exclude,
                make_blueprint=True,
                stop_depth=stop_depth,
                stop_keys=stop_keys
            )
            keys = [tuple(get_indices_from_blueprint(i, self._blueprint)) for i in range(len(self._blueprint))]
            super().__init__(dict(zip(keys, flattened_data)))

    @property
    def blueprint(self):
        return self._blueprint

    def structured(self):
        return construct(
            blueprint=self._blueprint,
            from_flat=list(self.data.values())
        )

    def aggregate(self, process_fn, process_types, include=None, exclude=None, *args, **kwargs):
        data = self.structured
        data = aggregate(data, process_fn, process_types, include, exclude, *args, **kwargs)
        flattened_data, self._blueprint = flatten(data, include=include, exclude=exclude, make_blueprint=True)
        keys = [tuple(get_indices_from_blueprint(i, self._blueprint)) for i in range(len(self._blueprint))]
        self.data = dict(zip(keys, flattened_data))

    def sub_nest(self, inner_key, depth_scope=-1):
        _sub_nest = Nest()
        _sub_nest.parent_nest = self
        _sub_nest._blueprint, _sub_nest.data = [], {}
        for i, (key_path, value) in enumerate(self.items()):
            if key_path[depth_scope] == inner_key:
                _sub_nest._blueprint.append(self._blueprint[i])
                _sub_nest.data[key_path] = value
        return _sub_nest

    def _maybe_parent_nest(self):
        if self.parent_nest:
            return self.parent_nest.data
        else:
            return self.data

    def __getitem__(self, key):
        if self.parent_nest:
            return self.parent_nest.__getitem__(key)
        else:
            return super(Nest, self).__getitem__(key)

    def __setitem__(self, key, item):
        if self.parent_nest:
            self.parent_nest.data[key] = item
        self.data[key] = item

    def __delitem__(self, key):
        if self.parent_nest:
            del self.parent_nest.data[key]
        del self.data[key]


def flatten(data, include=None, exclude=None, make_blueprint=False, stop_depth=None, stop_keys=None):

    default_flatten = set(include) if include else {dict, list, tuple, set, Nest}
    exclude = set(exclude) if exclude else set()
    flatten_types = default_flatten - exclude

    flattened = []
    foundation = (None, data.__class__)
    blueprint = []

    def _nest_iterator(subdata):
        if isinstance(subdata, Mapping):
            for index, item in subdata.items():
                yield index, item
        elif isinstance(subdata, Iterable):
            for index, item in enumerate(subdata):
                yield index, item

    if stop_keys and not isinstance(stop_keys, list):
        stop_keys = [stop_keys]

    def _maybe_exclude_from_flattening(_depth, index):
        if not stop_keys:
            return False
        _exclude = False
        for stop_key in stop_keys:
            if isinstance(stop_key, tuple):
                _exclude = _exclude or (_depth == stop_key[0] and index == stop_key[-1])
            else:
                _exclude = _exclude or index == stop_key
        return _exclude

    def _flatten(subdata, parent_path, depth):
        if isinstance(subdata, tuple(flatten_types)) and bool(subdata) and depth != stop_depth:
            depth += 1

            for index, item in _nest_iterator(subdata):
                nest_target = (index, item.__class__)
                nest_path = [c for c in parent_path]
                nest_path.append(nest_target)
                if not _maybe_exclude_from_flattening(depth, index):
                    _flatten(item, nest_path, depth)
                else:
                    flattened.append(item)
                    blueprint.append(nest_path)
        else:
            flattened.append(subdata)
            blueprint.append(parent_path)

    _flatten(data, [foundation], -1)

    if make_blueprint:
        return flattened, blueprint
    else:
        return flattened


def construct(blueprint, from_flat=None):
    make_foundation = blueprint[0][0][-1]
    structure = make_foundation()
    _from_flat = from_flat or [None] * len(blueprint)
    converters = []

    assert len(blueprint) == len(from_flat), 'Error: blueprint and from_flat are not the same length.'

    if len(blueprint) == 1 and len(blueprint[0]) == 1:
        return from_flat[0]

    def maybe_make_instance(class_candidate):
        try:
            return class_candidate()
        except Exception:
            return None

    def _is_maybe_empty(_nestling):
        try:
            return _nestling
        except Exception:
            return False

    for nest_path, nestling in zip(blueprint, _from_flat):

        current_level = structure
        parent_level = None
        parent_index = None

        for i, target in enumerate(nest_path[1:]):

            target_index, target_class = target

            if i != len(nest_path) - 2 or not _is_maybe_empty(nestling):
                obj = maybe_make_instance(target_class)
            else:
                obj = nestling

            try:  # maybe index or key already exists
                current_level.__getitem__(target_index)

            except KeyError:  # key did not already exist
                current_level.__setitem__(target_index, obj)

            except (IndexError, AttributeError):  # index did not already exist or __getitem__ not an attribute

                if hasattr(current_level, 'append'):  # in case we can append
                    current_level.append(obj)

                else:  # we can not append, so we should convert to list and later back
                    converters.append(nest_path[:i+1])
                    current_level = list(current_level)
                    current_level.append(obj)

                    if parent_level is not None and parent_index is not None:
                        parent_level[parent_index] = current_level
                    else:
                        structure = current_level

            parent_level = current_level
            parent_index = target_index
            if i != len(nest_path) - 2:
                current_level = current_level[target_index]

    converters = sorted(converters, key=lambda x: (-len(x), x))
    for converter in converters:  # convert the lists back to original type
        cur_level = structure

        if len(converter) == 1:
            conversion = converter[0][-1]
            structure = conversion(structure)

        else:
            for i, target in enumerate(converter[1:]):
                if i != len(converter) - 2:
                    cur_level = cur_level[target[0]]

                else:
                    conversion = target[-1]
                    cur_level[target[0]] = conversion(cur_level[target[0]])

    return structure


def aggregate(data, process_fn, process_types, include=None, exclude=None, *args, **kwargs):

    include = set(include) if include else {dict, list, tuple, set, Nest}
    exclude = set(exclude) if exclude else set()
    include = include - exclude

    def _nest_values(subdata):
        if isinstance(subdata, Mapping):
            return subdata.values()
        elif isinstance(subdata, Iterable):
            return subdata

    def _aggregate(subdata, process_fn, process_types, *args, **kwargs):
        if not isinstance(subdata, tuple(include)):
            return subdata

        if all(isinstance(item, process_types) for item in _nest_values(data)):
            return process_fn(list(_nest_values(subdata)), *args, **kwargs)

        if isinstance(subdata, Mapping):
            for key, value in subdata.items():
                subdata[key] = _aggregate(value, process_fn, process_types, *args, **kwargs)

        elif isinstance(subdata, Iterable):
            data_class = subdata.__class__
            subdata = list(subdata)
            for index, item in enumerate(subdata):
                subdata[index] = _aggregate(item, process_fn, process_types, *args, **kwargs)
            subdata = data_class(subdata)

        return subdata

    return _aggregate(data, process_fn, process_types, *args, **kwargs)


def get_element_by_indices(nested_list, indices):
    return reduce(lambda sub_list, index: sub_list[index], indices, nested_list)


def get_indices_from_blueprint(flat_index, blueprint):
    path = blueprint[flat_index]
    return tuple([c[0] for c in path[1:]])


def main():

    # Define the original data structure
    data = (
        'Egg',
        'Hatchling',
        'Chick',
        {
            'species': 'Blue Jay',
            'nest_location': {
                'latitude': 43.683573,
                'longitude': -79.635940
            },
            'siblings': [
                {'name': 'Sky', 'age': 5},
                {'name': 'Fluffy', 'age': 3}
            ],
            'food_preferences': (
                {'insects', 'berries', 'seeds'},
                ('worms', 'grubs', 'crickets')
            ),
        },
        Nest(
            ['Twigs', 'Leaves', 'Moss', {'feathers': 3, 'straw': 1}, {'pine needles', 'mud'}, (1, 2, 3)]
        ),
        None
    )

    # Print the original data structure
    print("Original data including a Nest:")
    print(data)

    # Flatten the data
    flat_data, blueprint = flatten(data, make_blueprint=True, stop_keys='nest_location')
    print("\nFlattened data:")
    print(flat_data)

    print("\nBlueprint:")
    print(blueprint)

    # Construct the nested data structure from the blueprint
    reconstructed_data = construct(blueprint, flat_data)
    print("\nReconstructed data:")
    print(reconstructed_data)

    # Test that the original data and reconstructed data are the same
    assert data == reconstructed_data, "Error: The original data and reconstructed data are not the same"

    # Create a Nest object from the original data
    nest = Nest(data)

    # Iterate through the Nest object using the items() method
    print("\nIterate data as a Nest:")
    for k, v in nest.items():
        print(f'key={k}, value={v}')

    # Print the Nest object
    print("\nStructured Nest:")
    print(nest)

    # Test that the original data and structured Nest are the same
    assert data == nest.structured(), "Error: The original data and the structured nest are not the same"

    # Aggregate all strings:
    aggregation = aggregate(data, lambda x: '\n'.join(x), process_types=str)
    print("\nAggregated data:")
    print(aggregation)

    # Not a data structure test:
    text = 'Just how python has their own nested structures, bird nests are also a ' \
           'form of nested architecture, with each layer of twigs and grass intricately ' \
           'woven together to form a cozy home for their feathered inhabitants.'
    flat_text, blueprint_text = flatten(text, make_blueprint=True)
    print("\nText:")
    print(text)
    print("\nText after flattened:")
    print(flat_text)
    print("\nText blueprint:")
    print(blueprint_text)
    print("\nReconstructed text:")
    print(construct(blueprint_text, flat_text))

    # Empty data test:
    empty_data = []
    flat_empty_data, blueprint_empty_data = flatten(empty_data, make_blueprint=True)
    print("\nEmpty data:")
    print(empty_data)
    print("\nEmpty data after flattened:")
    print(flat_empty_data)
    print("\nEmpty data blueprint:")
    print(blueprint_empty_data)
    print("\nReconstructed empty data:")
    print(construct(blueprint_empty_data, flat_empty_data))


if __name__ == '__main__':
    main()
