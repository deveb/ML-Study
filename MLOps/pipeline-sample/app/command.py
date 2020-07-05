from app import db
import click


@click.group()
def cli():
    pass

@click.command()
def init_db():
    click.echo('Initialized the database')
    import models
    db.create_all()

cli.add_command(init_db)


if __name__ == '__main__':
    cli()