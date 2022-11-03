import click
from speciesselector.database.management import mk_session
from speciesselector.database import orm


@click.command()
@click.option('--old', required=True)
@click.option('--new', required=True)
def main(old, new):
    engine_old, session_old = mk_session(old, new_db=False)
    engine_new, session_new = mk_session(new, new_db=True)
    for table in orm.Base.metadata.sorted_tables:
        old_objects = session_old.query(table)
        for obj in old_objects:
            session_new.execute(table.insert(obj))
        session_new.commit()


if __name__ == "__main__":
    main()

