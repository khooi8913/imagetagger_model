# Generated by Django 2.0.9 on 2018-12-10 06:11

import django.contrib.postgres.fields.jsonb
from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ('images', '0017_imageset_zip_state'),
    ]
    
    operations = [
        migrations.AddField(
            model_name='image',
            name='metadata',
            field=django.contrib.postgres.fields.jsonb.JSONField(default=dict),
        ),
        migrations.AddField(
            model_name='imageset',
            name='metadata',
            field=django.contrib.postgres.fields.jsonb.JSONField(default=dict),
        ),
    ]