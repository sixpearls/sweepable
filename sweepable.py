import peewee
from playhouse import reflection
from playhouse import migrate
import os
import sys
import inspect
import pickle
import copy
import datetime
from playhouse.sqliteq import SqliteQueueDatabase

__version__ = '0.0.1'

# TODO: this should have an option handler for the filename, database type, etc
# this should include ways to over-write the model, metaclass, and _meta class
data_root = os.path.join('..','data')
db_name = 'project_sweeps.db'
db = SqliteQueueDatabase(os.path.join(data_root, db_name))
VERBOSE_RUN = True
# TODO: the SqliteQueueDatabase seems like a good replacement for
# peewee.SqliteDatabase, but the asynch writing means the table creation 
# transaction may not be complete before the first query if not explicitly
# created.

# SqliteQueueDatabase seems to be the recommended way to avoid lock timeout
# for Sqllite:
# https://github.com/coleifer/peewee/issues/1071
# http://charlesleifer.com/blog/multi-threaded-sqlite-without-the-operationalerrors/

# TODO: need to figure out the best API for auto_save and auto_migrate

# TODO: CLI and/or interactive tools. Presumably independent of @sweepable use
# list tables (models)
# drop table by name
# drop tables not in file(s)/dirs?
# rename columns
# 


migrator = migrate.SqliteMigrator(db)

type_to_field = {
    int: peewee.IntegerField,
    float: peewee.FloatField,
    bool: peewee.BooleanField,
}

try:
    introspected_models = reflection.Introspector.from_database(db)\
        .generate_models(literal_column_names=True)
except: # TODO: determine what errors to catch??
    introspected_models = {}


class FileAccessor(peewee.FieldAccessor):
    def __get__(self, instance, instance_type=None):
        if instance is None:
            return self.field
        if self.field.name not in instance.__filedata__:
            if self.field.name in instance.__data__:
                instance.__filedata__[self.field.name] = \
                    self.field.read(instance)
            else:
                return None
        return instance.__filedata__[self.field.name]

    def __set__(self, instance, value):
        if value is None:
            return
        if isinstance(value, str):
            instance.__data__[self.field.name] = value
            instance.__filedata__[self.field.name] = self.field.read(instance)
        elif self.field.valid_value_type(value):
            instance.__filedata__[self.field.name] = value
            if instance.id is not None:
                instance.__data__[self.field.name] = \
                    self.field.get_total_path(instance)
        else:
            raise ValueError("Trying to assign invalid type to %s" % 
                self.field.name)
        instance._dirty.add(self.name)


def default_path_generator(field, model_instance):
    return os.path.join(data_root, field.model._meta.table_name, field.name)

def default_filename_generator(field, model_instance):
    return str(model_instance.id)

def pickle_reader(fname, field):
    with open(fname, 'rb') as buf:
        value = pickle.load(buf)
    return value

def pickle_writer(fname, field, value):
    with open(fname, 'wb') as buf:
        pickle.dump(value, buf)

# TODO: could also store the data_type in the DB as part of sweepable's meta tables
class FileField(peewee.CharField):
    accessor_class = FileAccessor
    def __init__(self, data_type=None, path_generator=default_path_generator, 
                 filename_generator=default_filename_generator,
                 reader=pickle_reader, writer=pickle_writer,):
        super().__init__(max_length=255, null=True) # TODO: settings?
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

    def valid_value_type(self, value):
        return isinstance(value, self.data_type) 

    def read(self, instance):
        value = self.reader(instance.__data__[self.name], self)
        if not self.valid_value_type(value):
            raise ValueError("Trying to read invalid type to %s" % 
                self.name)
        return value

    def write(self, instance, value):
        # this will update the pathname using the current path function,
        # TODO: may want to make an API for writing (this) and moving?
        if instance.id is None:
            raise ValueError("Cannot write %s because %s has no id" % 
                (str(self), str(instance)))
        instance.__data__[self.name] = self.get_total_path(instance)
        if not os.path.isdir(self.get_path(instance)):
            os.makedirs(self.get_path(instance))
        # TODO: I am not sure why but I was getting an error on this before.
        # possibly it was due to get_or_run not finding it in the database?
        # or files got saved before the output fields got written to DB?
        # """
        if os.path.exists(self.get_total_path(instance)):
            # TODO: a flag to allow over-writing?
            raise ValueError("File %s exists for %s" % 
                    (self.get_total_path(instance), 
                    '.'.join((str(instance), self.name)))
                )
        # """
        self.writer(self.get_total_path(instance), self, value)

    def delete(self, instance):
        if os.path.exists(self.get_total_path(instance)):
            os.remove(self.get_total_path(instance))

try:
    import numpy
except:
    numpy = None
else:
    def numpy_filename_generator(field, model_instance):
        return str(model_instance.id) + ".npy"

    def numpy_reader(fname, field):
        return numpy.load(fname, allow_pickle=False)

    def numpy_writer(fname, field, value):
        numpy.save(fname, value, allow_pickle=False)

    class NDArrayField(FileField):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, data_type=numpy.ndarray, 
                filename_generator=numpy_filename_generator, 
                reader=numpy_reader, writer=numpy_writer, **kwargs)

    type_to_field[numpy.ndarray] = NDArrayField

try:
    import pandas
except:
    pandas = None
else:
    def dataframe_filename_generator(field, model_instance):
        return str(model_instance.id) + ".csv"

    def dataframe_reader(fname, field):
        return pandas.read_csv(fname, index_col=0)

    def dataframe_writer(fname, field, value):
        value.to_csv(fname)

    class DataFrameField(FileField):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, data_type=pandas.DataFrame,
                filename_generator=dataframe_filename_generator, 
                reader=dataframe_reader, writer=dataframe_writer, **kwargs)


    type_to_field[pandas.DataFrame] = DataFrameField

    def to_dataframe(query):
        sql, params = query.sql()
        return pandas.read_sql(sql=sql, con=db.connection(), params=params)


"""
the full cascade of Metadata, ModelBase, and Model subclasses may not be 
necessary to handle the FileFields, but I am creating them now in case I need
to hook into the machinery later
"""
class SweepableMetadata(peewee.Metadata):
    def __init__(self, *args, **kwargs):
        self.filefields = {}
        self.nonfilefields = {}
        super().__init__(*args, **kwargs)


class SweepableModelBase(peewee.ModelBase):
    def __new__(cls, name, bases, attrs):
        cls = super().__new__(cls, name, bases, attrs)
        for key, value in cls._meta.fields.items():
            if isinstance(value, FileField):
                cls._meta.filefields[key] = value
            else:
                cls._meta.nonfilefields[key] = value
        return cls

    def __model_str__(self): # TODO: I'm not sure why I can't overwrite __str__
        return '.'.join(self.__name__.split('__'))

    def __repr__(self):
        return '<SweepableModel: %s>' % self.__model_str__()

class SweepableModel(peewee.Model, metaclass=SweepableModelBase):
    start_time = peewee.DateTimeField()
    stop_time = peewee.DateTimeField()

    class Meta:
        database = db
        legacy_table_names = False
        model_metadata_class = SweepableMetadata

    def __init__(self, *args, **kwargs):
        self.__filedata__ = {}
        super().__init__(*args, **kwargs)
        if self.id is None:
            self.sweeper.unsaved_instances.append(self)

    def save_run(self, force_insert=False, only=None):
        if not only:
            only = self.__data__.copy()

        dirty_filefields = []
        for filefield in self._meta.filefields:
            if filefield in self._dirty:
                dirty_filefields.append(filefield)

        save_file_fields = False

        if self.id is None:
            # raise error on no id and no force_insert?
            # TODO: save only= prune(only, non-file) to get ID
            save_file_fields = True
            nonfile_only = self._prune_fields(only, 
                                self._meta.nonfilefields)
            saveval = super().save(force_insert, nonfile_only)
            # then save only=prune(only, file) or really fields that may/don't
            # depend on ID
            if not self.id:
                raise ValueError("Could not obtain an ID, try setting " +
                    "force_insert=True")
        else:
            saveval = super().save(force_insert, only)

        # TODO: Is there a way make writing files tied to transaction?
        for filefield in dirty_filefields:
            self._meta.fields[filefield].write(self, getattr(self, filefield))

        if save_file_fields:
            file_only = self._prune_fields(only, 
                                self._meta.filefields)
            saveval = super().save(force_insert, file_only)

        if self in self.sweeper.unsaved_instances:
            self.sweeper.unsaved_instances.remove(self)
        return saveval

    def delete_run(self, recursive=False, delete_nullable=False):
        for filefield in self._meta.filefields:
            self._meta.fields[filefield].delete(self)
        return super().delete_instance(recursive, delete_nullable)
    
    def __str__(self):
        return ', '.join([
            '%s=%s' % (field, getattr(self, field)) 
            for field in self.sweeper.input_fields])
    
    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__model_str__(), self.__str__())

    def outputs(self):
        return tuple([getattr(self,key) for key in self.sweeper.output_fields])

    # TODO: __str__ here gets used in ModelBase __repr__ assignment :5419
    # TODO: is there some way to prevent saving when not creating?

class sweeper(object):
    def __init__(self, function, output_fields, auto_migrate=False, 
    save_on_run=True, create_table=True, delete_files=True):
        self.function = function
        self.signature = inspect.signature(self.function)
        self.name = function.__code__.co_name
        self.module = os.path.basename(function.__code__.co_filename)\
            .split('.py')[0]

        # TODO: figure out how to incorporate class name if it's a method? 
        # Or disallow? API for over-writing naming scheme?

        self.model = None
        self.unsaved_instances = [] 
        # TODO: for some reason testing set membership failed but list worked?
        self.input_fields = {}
        self.output_fields = output_fields
        self.auto_migrate = auto_migrate
        self.save_on_run = save_on_run
        self.create_table = create_table
        self.delete_files = delete_files

        # TODO: is there a reason not to validate here by default? see 
        # discussion about queue database above

    def process_signature(self):
        # Should this be DRYd up w.r.t. the output field processing? No, 
        # different enough logic
        for param in self.signature.parameters:
            param_default = self.signature.parameters[param].default
            
            if isinstance(param_default, FileField):
                raise ValueError("I don't know how to handle this yet")
            elif (isinstance(param_default, peewee.Field) and 
            param_default.default is not None):
                self.input_fields[param] = param_default
            elif isinstance(param_default, sweeper):
                self.input_fields[param] = peewee.ForeignKeyField(
                                                    param_default.model)
            elif type(param_default) in type_to_field:
                self.input_fields[param] = type_to_field[type(param_default)](
                                                    default=param_default)
            else:
                raise ValueError("Unknown field for input argument")

    def depends_on(self, *args):
        # TODO: this is a placeholder for a non-"framework" usage of sweepable
        return

    def get_add_drop_fields(self):
        # TODO: this feels like a generic function, not a method on sweeper?
        # are there any other methods that ought to be refactored?
        old_field_set = set(introspected_models[self.model.__name__]\
                ._meta.fields.values())
        new_field_set = set(self.model._meta.fields.values())

        # TODO: other checks for equality? Peewee hash is only based on 
        # field & model name (does python include type?) If PeeWee does 
        # defaults on python side, then PW's hash + field type would be 
        # sufficient. Otherwise, may want to update default. If we 
        # eventually have a table for metadata, will need to check that.
        drop_fields = old_field_set - new_field_set
        add_fields = new_field_set - old_field_set

        for drop_field in drop_fields.copy():
            if isinstance(drop_field, peewee.ForeignKeyField):
                mfield = copy.copy(drop_field)
                mfield.name = mfield.name.replace('_id', '')
                if mfield in add_fields:
                    drop_fields.remove(drop_field)
                    add_fields.remove(mfield)

        return add_fields, drop_fields

    def migrate(self, add_fields, drop_fields):
        # TODO: make transaction. return status flag?
        migrate.migrate(
            *tuple([migrator.drop_column(
                table=field.model._meta.table_name,
                column_name=field.column_name,
                ) for field in drop_fields] +
              [migrator.add_column(
                table=field.model._meta.table_name,
                column_name=field.column_name,
                field=field,
                ) for field in add_fields])
        )

    def validate(self, do_migrate=False):
        self.process_signature()
        arg_fields = {**self.input_fields, **self.output_fields}
        arg_fields['__repr__'] = SweepableModel.__repr__
        # TODO: this is a bug peewee, __repr__ can't be inherited.
        self.model = type(
            '%s__%s' % (self.module, self.name), (SweepableModel,), arg_fields)
        self.model.sweeper = self
        self.model.__module__ = self.module

        for field in self.input_fields:
            self.input_fields[field] = self.model._meta.fields[field]

        for field in self.output_fields:
            self.output_fields[field] = self.model._meta.fields[field]

        # add the SweepableModel class to the module where the sweepable
        # function is defined so that instances can be pickled

        try:
            setattr(sys.modules[self.module], self.model.__name__, self.model)
        except: # TODO: is it good to silently fail?
            pass

        # add all the model field's to the sweeper to provide a shortcut for
        # queries 
        for field_name, field in self.model._meta.fields.items():
            setattr(self, field_name, field)

        if self.model.__name__ in introspected_models:
            add_fields, drop_fields = self.get_add_drop_fields()

            if not (self.auto_migrate or do_migrate) and \
            (drop_fields or add_fields):
                error_string = self.model.__model_str__() + " current code " +\
                "specification does not match database. You may need to" +\
                " migrate the database."
                if drop_fields:
                    error_string += "\nNon-matching fields in database: " +\
                        ", ".join([str(el) for el in drop_fields])
                if add_fields:
                    error_string += "\nNon-matching fields in code spec: " +\
                        ", ".join([str(el) for el in add_fields])

                raise ValueError(error_string)
            else:
                self.migrate(add_fields, drop_fields)
        
        elif self.create_table:
            self.model.create_table()

    def drop_table(self):
        db.drop_tables(self.model)

    def select_or_run(self, *args, **kwargs):
        """
        This will be slower than a query if you are only interested in runs
        that have already been completed, but this will make sure all points in
        the run matrix are returned, running them if needed.
        """
        query_rows = self.bind_signature(args, kwargs)

        model_instances = []
        for query_row in query_rows:
            model_instances.append(self.get_or_run(**query_row))

        if len(model_instances) == 1: # TODO: Is this actually a good API?
            return model_instances[0]
        return model_instances

    def bind_signature(self, args, kwargs):
        # TODO: If we require kwargs, we could possibly skip binding the 
        # signature and use the model instantiation? I guess that doesn't solve
        # broadcasting binding either. Can we just overwrite .isin type query
        # fields?

        # TODO: helpful checks for binding args and kwargs? maybe type checks?
        # I tried to pass in by args only, and missed an arg and got a weird
        # error because a default sweeper was used. So I guess also update the
        # signature to use an instance of the sweepable model instead of the
        # sweeper?

        bound_args = self.signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        num_rows = 1
        varying_fields = []
        static_fields = {}
        
        for arg, val in bound_args.arguments.items():
            # TODO: handle field types that may have __len__
            # TODO: type checking on arguments
            arg_rows = getattr(val, '__len__', lambda: 1)()
            if arg_rows > 1 and arg_rows != num_rows:
                if num_rows == 1:
                    num_rows = arg_rows
                else:
                    # TODO: if multiple parameters are given multiple values, 
                    # should we raise an error or do the tensor product? I guess
                    # this could be a setting switch?
                    raise ValueError("Could not broadcast arguments to call")
            if arg_rows > 1:
                varying_fields.append(arg)
            elif hasattr(val, '__len__') and len(val) == 1:
                static_fields[arg] = val[0]
            else:
                static_fields[arg] = val

        query_rows = []
        for row in range(num_rows):
            query_rows.append(static_fields.copy())
            for field in varying_fields:
                query_rows[-1][field] = bound_args.arguments[field][row]

        return query_rows

    def get_or_run(self, *args, **kwargs):
        # returns the model instance
        query_rows = self.bind_signature(args, kwargs)
        if len(query_rows) > 1:  
            raise ValueError("Calling get_or_run with more than one set of " +
                "parameters. Consider using select_or_run")
        else:
            query_row = query_rows[0]
        # returns the SweepableModel instance(s) associated with call(s)
        query = self.model.select()
        for field, value in query_row.items():
            query = query.where(getattr(self.model, field) == value)

        try:
            instance = query.get()
        except (self.model.DoesNotExist, peewee.OperationalError):
            instance = self.model(**query_row)
            do_run = True
        else:
            do_run = False
            for field in self.output_fields:
                if getattr(instance, field, None) is None:
                    do_run = True
        if do_run:
            if VERBOSE_RUN:
                print("running " + str(instance))
            self.run_instance(instance)
        else:
            if VERBOSE_RUN:
                print("skipping " + str(instance))
        return instance

    def __call__(self, *args, **kwargs):
        # returns the model instance

        # I strongly prefer working with the SweepableModel instances.
        # TODO: add a method/attribute on the model to return original outputs
        instance = self.get_or_run(*args, **kwargs)
        return instance

    def run_instance(self, instance, do_save=False):
        # TODO: use context manager to explicitly do DB dis/connect? AND tie 
        # file writing to transaction so it can be rolled back on error?
        
        instance.start_time = datetime.datetime.now()
        kwargs = {fname: getattr(instance, fname) 
            for fname in self.input_fields}
        result = self.function(**kwargs)
        instance.stop_time = datetime.datetime.now()

        # TODO: figure out better way to determine matching lengths
        result_length = getattr(result, '__len__', lambda: 1)()
        if len(self.output_fields) == 1 and result_length != 1:
            result = [result]
        if len(self.output_fields) != len(result):
            raise ValueError(str(self) + " has mismatch between defined " + 
                "output fields and returned values for " + str(instance))
        for output_field, value in zip(self.output_fields, result):
            setattr(instance, output_field, value)

        if self.save_on_run or do_save:
            instance.save_run()
        else:
            self.unsaved_instances.append(instance)
        return instance

    def __repr__(self):
        return "<Sweeper for %s.%s>" % (self.module, self.name)

    @property
    def model(self):
        if self._model is None:
            self.validate()
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

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

    if pandas:
        def to_dataframe(self, query=None):
            if query is None:
                query = self.select()
            return to_dataframe(query)


class sweepable(object):
    def __init__(self, **kwargs):
        output_fields = {}
        for arg in kwargs:
            arg_type = kwargs[arg]
            if isinstance(arg_type, peewee.Field):
                output_fields[arg] = arg_type
            elif issubclass(arg_type, peewee.Field):
                output_fields[arg] = arg_type()
            else:
                output_fields[arg] = type_to_field[arg_type]()
            # TODO: better way to enforce nullable column with null default?
            output_fields[arg].null = True
        self.output_fields = output_fields

    def __call__(self, function):
        return sweeper(function, self.output_fields)

class sweepable_test(sweepable):
    def __call__(self, function):
        return sweeper(function, self.output_fields,
            auto_migrate=True, create_table=False, save_on_run=False)