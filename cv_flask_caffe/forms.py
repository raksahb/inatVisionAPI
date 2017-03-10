from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from wtforms import StringField

class ImageForm(FlaskForm):
    path = StringField('path')
    version = StringField('version')
    image = FileField('image',
        validators=[
            FileRequired(message="Please include 'image' field.")
        ])
