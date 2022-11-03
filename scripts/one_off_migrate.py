import click
import sqlalchemy.exc

from speciesselector.database.management import mk_session
from speciesselector.database import orm
import sqlalchemy

Base = sqlalchemy.ext.declarative.declarative_base()


@click.command()
@click.option('--old', required=True)
@click.option('--new', required=True)
def main(old, new):
    engine_old, session_old = mk_session(old, new_db=False)
    engine_new, session_new = mk_session(new, new_db=True)
    old_metadata = Base.metadata
    old_metadata.reflect(engine_old)
    for otab, table in zip(old_metadata.sorted_tables, orm.Base.metadata.sorted_tables):
        # this does not handle adding columns with a value to existing tables!
        # because all that was needed when this was jotted down was adding to empty tables
        # and / or a bunch of 'None' values
        # if more is necessary, it is probably time to get out a real migration tool
        old_objects = session_old.query(otab)
        for obj in old_objects:
            session_new.execute(table.insert(obj))

        session_new.commit()


if __name__ == "__main__":
    main()

