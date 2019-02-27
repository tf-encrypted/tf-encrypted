# TODO should these really be classes? Dict? Function?
class Add:
    name = "Add"
    num_inputs = 2
    num_outputs = 1
    attrs = []


class Sub:
    name = "Sub"
    num_inputs = 2
    num_outputs = 1
    attrs = []


class Mul:
    name = "Mul"
    num_inputs = 2
    num_outputs = 1
    attrs = []


# TODO Encode? Convert? Cast?
class Cast:
    name = "Cast"
    num_inputs = 2
    num_outputs = 1
    attrs = ["players"]


ops_list = [Add, Sub, Mul, Cast]
