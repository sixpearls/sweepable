import peewee as pw # Lorena Barba doesn't like aliasing imports, what do you think?
import os
import inspect
import pickle

__version__ = '0.0.1'

# this should have an option handler for the filename, database type, etc
data_root = os.path.join('..','data')
db_name = 'project_sweeps.db'
db = pw.SqliteDatabase(os.path.join(data_root, db_name))
type_to_field = {
    int: pw.IntegerField,
    float: pw.FloatField,
}

"""
Management commands:
reset model (drop table, possibly remove filefield data)
migrate - limited use case, update schema and possibly re-run outputfields?
if doing git checking, 
"""


"""
Prototype using Pyzo (MATLAB-like) or Jupyter notebooks (Mathematica-like),
seems very natural especially for users who started with MATLAB and/or 
Mathematica. If you don't know how things should behave, create chunks that are
run-able, keep a persistent workspace incrementally test code for each portion
of the simulation process (different references/other setup, main run, 
performance evaluation would be a common 3-step simulation pipeline in
controls).

It is good to get in a habit to keep parameters to the top of (sections or 
cells of) code for each section. Then they can be easily adapted to sweepable
functions. Then get in the habit of breaking out each step in the simulation
process (i.e., the same reference, run, evaluate or other appropriate
"pipeline") -- 

references should be simulated once, then loaded (or call a sweepable reference
generator)
analysis could call a sweepable reference 

Once a sweepable function, use ORM-like API to more easily analyze (build 
summary tables, make plots, etc)

Also, runners should generate new objects rather than persistent reference in
module as much as possible. I think this will make it easier to convert to
distributed computing.
sweepable makes it easier to implement the practice of never running a
simulation and then doing something with it in the same namespace.

all sweepable functions should assume they run 1-at-a-time. I think this makes
the API easier, and I assume you wouldn't need this if it were ufuncable or 
something. I guess we could provide some kind of hooks for a batch-processable
numerical experiment step, not sure. between caching and/or distributed
computing, and most use-cases not being ammenable anyway, this should allow
good performance and clean writing for the user.

make input_default a sweepable object so sweepable knows you know.
you can avoid copying parameter names that way, but then probably can only call
using queries? or the object returned by a get?


Could you make a sweepable aware objects wear the default is a partial query? 
You would have to be deferred somehow but it could be a requirements for this 
setting function, where a different function might require it subset of 
something in the compliment

Should I do any magic of stripping out either repeated argument names or double
underscore argument tracing to just rely on the foreign key? This might be 
necessary to really do the double underscore routing for field queries for a 
non-sweepable-aware function. This would also allow reverse queries, to find
all (then filter) sim results based on this reference sim.
"""


"""
peewee things:

Use new table names: Use legacy_table_names false

File Fields
Do Ineed to assign an ``accessor_class`` to the Field? assign dunder get/set
Field should need db_value and python_value

ForeignKeyField is probably closest built-in that matches file pattern
db_value essentially returns the PK of the model instance
python_value 


"""

class FileAccessor(pw.FieldAccessor):
    def __get__(self, instance, instance_type=None):
        if instance is None:
            return self.field
        if self.field.name not in instance.__filedata__:
            instance.__filedata__[self.field.name] = self.field.read(instance)
        return instance.__filedata__[self.field.name]

    def valid_value_type(self, value):
        return isinstance(value, self.field.data_type)

    def __set__(self, instance, value):
        # TODO: is there a way to protect this except for a get_or_create call?
        # TODO: should also try to mark objects as immutable?
        # TODO: type checking based on field.data_type
        if not self.valid_value_type(value):
            raise ValueError("Trying to assign invalid type to ")
        instance.__filedata__[self.field.name] = value
        self.field.write(instance, value)

def default_path_generator(field, model_instance):
    return os.path.join(data_root, field.model._meta.table_name, field.name)

def default_filename_generator(field, model_instance):
    return str(model_instance.id)

def pickle_reader(buffer, field):
    return pickle.load(buffer)

def pickle_writer(buffer, field, value):
    pickle.dump(value, buffer)

# TODO: If we actually store filepath in DB, we could allow changes in path 
# generation and still access files created in old schema
# TODO: could also store the data_type in the DB, but not sure how to add
# a single field that spans two columns
class FileField(pw.CharField):
    accessor_class = FileAccessor
    def __init__(self, data_type=None, path_generator=default_path_generator, 
                 filename_generator=default_filename_generator,
                 reader=pickle_reader, writer=pickle_writer,):

        if not inspect.isclass(data_type):
            raise ValueError("Generic FileField requires a data_type")
        self.data_type = data_type
        self.path_generator = path_generator
        self.filename_generator = filename_generator
        self.reader = reader
        self.writer = writer

    def get_path(self, instance):
        return self.path_generator(self, instance)

    def get_filename(self, instance):
        return self.filename_generator(self, instance)

    def get_total_path(self, instance):
        return os.path.join(self.get_path(instance),
                            self.get_filename(instance))

    def read(self, instance):
        # TODO: type checking, return a default type on any kind of error
        with open(self.get_total_path(instance), 'rb') as buffer:
            value = self.reader(buffer, self)
        return value


    def write(self, instance, value):
        if not os.path.isdir(self.get_path(instance)):
            os.makedirs(self.get_path(instance))
        with open(self.get_total_path(instance), 'wb') as buffer:
            self.writer(buffer, self, value)

    def __repr__(self):
        return "<FileField %s on %s>" % (self.__class__.__name__, self.model)

"""
the full cascade of Metadata, ModelBase, and Model subclasses may not be 
necessary to handle the FileFields, but I am creating them now in case I need
to hook into the machinery later
"""
class SweepableMetadata(pw.Metadata):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filefields = {}

    def add_filefield(self, field_name, filefield, set_attribute=True):
        if field_name in self.fields or field_name in self.filefields:
            raise ValueError("Duplicate field name for FileField")
        self.filefields[field_name] = filefield
        filefield.bind(self.model, field_name, set_attribute)


class SweepableModelBase(pw.ModelBase):
    def __new__(cls, name, bases, attrs):
        cls = super().__new__(cls, name, bases, attrs)
        for key, value in cls.__dict__.items():
            if isinstance(value, FileField):
                cls._meta.add_filefield(key, value)
        return cls

    def __repr__(self):
        return '<Sweepable: %s>' % self.__name__

class SweepableModel(pw.Model, metaclass=SweepableModelBase):
    class Meta:
        database = db
        legacy_table_names = False
        model_metadata_class = SweepableMetadata

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__filedata__ = {}

    # TODO: is there some way to prevent saving that isn't creating?
    def to_csv(self):
        # TODO: SweepableModel should provide at least a csv export method
        return

class sweeper(object):
    def __init__(self, function, output_fields, istest=False):
        self.function = function
        self.signature = inspect.signature(self.function)
        self.name = function.__name__ # or __qualname__?
        self.module = function.__module__ 
        print(self.module)
        # TODO: how to get filename instead of __main__?
        self.model = None
        self.input_fields = {}
        self.output_fields = output_fields
        self.istest = istest

        self.process_signature()
        self.validate()

    def process_signature(self):
        # Should this be DRYd up w.r.t. the output field processing?
        for param in self.signature.parameters:
            param_default = self.signature.parameters[param].default

            if (isinstance(param_default, pw.Field) and 
            param_default.default is not None):
                self.input_fields[param] = param_default
            elif isinstance(param_default, sweeper):
                self.input_fields[param] = pw.ForeignKeyField(param_default.model)
            elif isinstance(param_default, FileField):
                raise ValueError("I don't know how to handle this yet")
            elif type(param_default) in type_to_field:
                self.input_fields[param] = type_to_field[type(param_default)](
                    default=param_default)
            else:
                raise ValueError("Unknown field for input argument")

    def depends_on(self, *args):
        # TODO: this is a placeholder for a non-"framework" usage of sweepable
        return

    def validate(self):
        """
        [ ] check if table exists and if so, matches current fields
        [ ] eventually, would also check that the code hasn't changed -- would
        need to track pip (for all dependency versions) and git (for current 
        research project and all dependencies installed with -e). It would be
        really nice if it could check what files (or even classes/functions)
        have been changed and only enforce compatability for that.
        --- check that fieldnames __ tracking has a depends_on, these can
        have no default or None default --- actually, I want the original sweep-
        able functions to be totally agnostic to sweepable API. Adding a
        depends_on makes dot-able namespace routing do more "external" code like
        plotting. Especially with caching, (does that require somehow making 
        these objects more persistent? or will they always be persistent 
        enough?) it should be performant enough.

        code changing checks would also have to have a table of sweepable 
        metadata. then check that table in DB for sweepable matches current
        definition. possibly complicated for the file fields.

        [ ] for non framework usage, need every function that might hit the
        database to validate

        """
        arg_fields = {**self.input_fields, **self.output_fields}
        self.model = type(self.name, (SweepableModel,), arg_fields)
        # TODO: if istest, inspect table and drop if doesn't match
        # TODO: else: warn if doesn't match
        # TODO: allow a migrate migrate for non-matching, re-run to fill values
        self.model.create_table()

    def get_or_run(self, *args, **kwargs):
        return self.__call__(*args, **kwargs) # or self(*args, **kwargs)?
    
    def __repr__(self):
        return "<sweepable.sweeper for %s.%s>" % (self.module, self.name)

    def __call__(self, *args, **kwargs):
        """
        originally, I was thinking would be very nice to have object-dot-able
        tracking through foreign keys, but this only has to be accessible if
        you know you're using sweepable (aka, plotting swept results). The 
        general runner functions that can be made sweepable don't need to know
        about this -- assume they're just standard callables.

        But it might be nice if we could provide hooks for caching. And to stay
        dask compatible, for eventuality. caching for "get" calls should be as
        performant as building in sweep awareness from the beginning.

        this should be thought of as a get_or_create function from an ORM.
        should this even handle query's? or should that be in a separate method
        then break out the "call_function" logic?

        __call__ and get_or_run: try to broadcast to create rows of the DB

        then you can directly access select, get, get_or_none, get_by_id, and
        filter of the pw.Model from the sweepable
        """
        
        bound_args = self.signature.bind(*args, **kwargs)
        # TODO: allow get_or_create instead of foreignkey model instance? 
        bound_args.apply_defaults()
        num_rows = 1
        varying_fields = []
        static_fields = {}
        for arg, val in bound_args.arguments.items():
            # TODO: handle field types that may have __len__
            # TODO: type checking on arguments
            arg_rows = getattr(val, '__len__', 1) # could break if ndarray?
            if arg_rows > 1 and arg_rows != num_rows:
                if num_rows == 1:
                    num_rows = arg_rows
                else:
                    # TODO: if multiple parameters are given multiple 
                    # parameters, should we raise an error or do the tensor 
                    # product? I guess this could be a setting switch?
                    raise ValueError("Could not broadcast arguments to call")
            if arg_rows > 1:
                varying_fields.append(arg)
            else:
                static_fields[arg] = val

        query_rows = []
        for row in range(num_rows):
            query_rows.append(static_fields.copy())
            for field in varying_fields:
                query_rows[-1][field] = bound_args.arguments[field][row]

        model_instances = []
        for query_row in query_rows:
            instance, needs_to_run = self.model.get_or_create(**query_row)
            # TODO: should we write our own get_or_create logic?
            # having the id set before the filefields are created seems nice for
            # current implementation, but maybe that's wrong anyway and files
            # should only be saved once the record is created?
            if needs_to_run:
                # TODO: include a created_on and completed_on in SweepableModel?
                # which could be part of the default gen_pathname
                # and would also create some useful metadata

                # TODO: use constext manager to explicitly do DB dis/connect
                # AND don't create the instance until function runs and output
                # fields are assigned (or roll back on error)
                result = self.function(**query_row)
                # TODO: need to a binding for the output_fields, 
                # if single result and is an iterator, this fails
                for output_field, value in zip(self.output_fields, result):
                    setattr(instance, output_field, value)
                instance.save()
            model_instances.append(instance)
        if num_rows == 1:
            return instance
        return model_instances

    def select(self, *fields):
        return self.model.select(*fields)

    def get(self, *query, **filters):
        return self.model.get(*query, **filters)

    def get_or_none(self, *query, **filters):
        return self.model.get_or_none(*query, **filters)

    def get_by_id(self, pk):
        return self.model.get_by_id(pk)

    def filter(self, *dq_nodes, **filters):
        return self.model.filter(*dq_nodes, **filters)



"""
could we somehow make a __call__ so that the user code is

import sweepable

@sweepable(**output_fields)
def func_name(arg1=arg1_defualt, ...):
    ...
    return

?


ideal user code:
```
import sweepable
import pandas as pd
import numpy as np

@sweepable(ref_traj=np.array)
def get_reference(ref_param1=0., ...):
    ...
    return x_ref # allow tuple or dictionary?

@sweepable(out_traj=pd.DataFrame)
def sim_system(ref_param1=0., ..., sim_param1=0., sim_param2=0., ...):
    ref = get_reference(ref_param1, ...)
    ...
    return df
sim_system.depends_on(get_reference)

@sweepable(metric1=float, metric2=float, metric3) #default is single float?
def evaluate_sim(ref_param1=0., ..., sim_param1=0., sim_param2=0., ...):
    traj = sim_system(ref_param1, ..., sim_param1, sim_param2, ...)
    ref = get_reference(ref_param1, ...)
    ...
    return metrci1, metric2, metric3

evaluate_sim.depends_on(sim_system)
```

is there a way to avoid copying the signature if it's exactly the same??
I guess this would be the negative to making it broken-out into functions is
inherently repeating some persistent information. I guess we could make the
evaluator aware of sweepable?

```
@sweepable(ref_traj=np.array)
def get_reference(ref_param1=0., ...):
    ...
    return x_ref # allow tuple or dictionary?

@sweepable(out_traj=pd.DataFrame)
def sim_system(reference=get_reference, sim_param1=0., sim_param2=0., ...):
    ref = reference.ref_traj
    ...
    return df

@sweepable(metric1, metric2, metric3)
def evaluate_sim(sim_result=sim_system):
    # now sweepable knows you know, so user code can use orm-y dottable behavior
    traj = sim_result.out_traj 
    ref = sim_result.get_reference.ref_traj
    ...
    # should still assume 1-at-a-time calls. let sweepable build the table.
    return metrci1, metric2, metric3
# evaluate_sim.depends_on(sim_system) not needed because implied by default
```

example analysis code:
``` # Make a plot!
import matplotlib.pyplot as plt
fig, axes = plt.subplots(...)
labels = []
for sim in evaluate_sim.select(ref_param1__in=[], **query):
    sim.out_traj.plot(ax=axes, subplots=True, color=...)
    labels.append('param2 = {.2f}'.format(sim.param2))

axes[0].legend(labels)
```

apparently, I should provide a tool to cast a query as a dataframe?

```
metadata = evaluate_sim.select(sim_result__param3__in=[],...)
df = metadata._to_df() # ???
df.groupby(sim_result__param3).agg({'mean', 'median', 'stdev'})
```
"""

class sweepable(object):
    """
    it is conceivable that the same exact function could be used to in multiple
    pipelines of sweepable functions.

    you could create a wrapper function for each "pipeline" so it would have its
    own table and "connections." To make this easier, it would be nice if we
    could help copy and modify a signature, to DRY up this use case.

    we could also have a non-decorator call, like 
        func_name = sweepable(**output_fields)(func_name)
    actually, would that just work?

    """

    def __init__(self, **kwargs):
        output_fields = {}
        for arg in kwargs:
            arg_type = kwargs[arg]
            if isinstance(arg_type, (pw.Field, FileField)):
                output_fields[arg] = arg_type
            elif issubclass(arg_type, (pw.Field, FileField)):
                output_fields[arg] = arg_type()
            else:
                output_fields[arg] = type_to_field[arg_type]()
        self.output_fields = output_fields

    def __call__(self, function):
        return sweeper(function, self.output_fields)

class sweepable_test(sweepable):
    def __call__(self, function):
        return sweeper(function, self.output_fields, istest=True)