from flask_wtf import FlaskForm
from wtforms import StringField
from flask_wtf.file import FileField, FileRequired

class ImageForm( FlaskForm ):
    observation_id = StringField( "observation_id" )
    image = FileField( "image" )
    format = StringField( "format" )
