# Generated by Django 5.0.4 on 2024-10-23 22:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('simapp', '0023_plant_b'),
    ]

    operations = [
        migrations.AddField(
            model_name='datamodeloutput',
            name='profit',
            field=models.FloatField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='datamodeloutput',
            name='rain',
            field=models.FloatField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='datamodeloutput',
            name='temperature',
            field=models.FloatField(default=0),
            preserve_default=False,
        ),
    ]