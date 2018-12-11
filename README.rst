Sweepable
=========

Sweepable is a framework for taking down numerical/computational experiments
efficiently and productively. Sweepable is designed to help implement a work
flow that 


Workflow
--------
1. Prototype 
Using Pyzo (MATLAB-like) or Jupyter notebooks (Mathematica-like) seems very
natural especially for users who started with MATLAB and/or Mathematica. If you
don't know how things should behave, these tools allow you to create
chunks that are run-able while keeping a persistent workspace to incrementally
test code for each portion of the experiment. For control systems, it is 
common to generate different reference trajectories, simulate the main control
system, then performance evaluation.

2. Refine
It is good to get in a habit to keep parameters to the top of (sections or 
cells of) code for each section.


Using Sweepable
---------------

Then they can be easily adapted to sweepable
functions. Then get in the habit of breaking out each step in the simulation
process (i.e., the same reference, run, evaluate or other appropriate
"pipeline") -- 


Example user code
=================

Generic API definition:

.. code-block :: python

import sweepable

@sweepable(**output_fields)
def func_name(arg1=arg1_defualt, ...):
    ...
    return




Current target: Sweepable as a Framework

.. code-block :: python

    @sweepable(ref_t=np.array, ref_y)
    def get_reference(ref_param1=0., ...):
        ...
        return t_ref, x_ref # allow tuple or dictionary?

    @sweepable(out_traj=pd.DataFrame)
    def sim_system(reference=get_reference, sim_param1=0., sim_param2=0., ...):
        ref_t = reference.ref_t
        ref_y = reference.ref_y
        ...
        return (df,)

    @sweepable(metric1=float, metric2=float, metric3=float)
    def evaluate_sim(sim_result=sim_system):
        traj = sim_result.out_traj 
        ref_y = sim_result.get_reference.ref_y
        ...
        return metrci1, metric2, metric3

Which would let you do things like make a plot

.. code-block :: python

    # Make a plot!
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(...)
    labels = []
    for sim in evaluate_sim.select(ref_param1__in=[], **query):
        sim.out_traj.plot(ax=axes, subplots=True, color=...)
        labels.append('param2 = {.2f}'.format(sim.param2))

    axes[0].legend(labels)

Or generate some tables

.. code-block :: python

    metadata = evaluate_sim.select(sim_result__param3__in=[],...)
    df = metadata._to_df() # ???
    df.groupby(sim_result__param3).agg({'mean', 'median', 'stdev'})


Eventually, it might be nice if the code could be less coupled, as in

.. code-block :: python

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

    @sweepable(metric1=float, metric2=float, metric3=float)
    def evaluate_sim(ref_param1=0., ..., sim_param1=0., sim_param2=0., ...):
        traj = sim_system(ref_param1, ..., sim_param1, sim_param2, ...)
        ref = get_reference(ref_param1, ...)
        ...
        return metrci1, metric2, metric3
    evaluate_sim.depends_on(sim_system)


The ``depends_on`` would tell Sweepable that repeated arguments define a relationship between the functions.


Development notes
=================
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



it is conceivable that the same exact function could be used to in multiple
pipelines of sweepable functions.

you could create a wrapper function for each "pipeline" so it would have its
own table and "connections." To make this easier, it would be nice if we
could help copy and modify a signature, to DRY up this use case.

we could also have a non-decorator call, like 
    `func_name = sweepable(**output_fields)(func_name)`
actually, would that just work?


is there a way to avoid copying the signature if it's exactly the same??
I guess this would be the negative to making it broken-out into functions is
inherently repeating some persistent information. I guess we could make the
evaluator aware of sweepable?


Management commands?
--------------------
reset model (drop table, possibly remove filefield data)
migrate - limited use case, update schema and possibly re-run outputfields?
if doing git checking,



__call__ api
------------





