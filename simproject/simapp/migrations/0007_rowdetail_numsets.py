# Generated by Django 5.0.4 on 2024-10-03 18:47

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('simapp', '0006_rename_input_data_simulation_input_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='rowdetail',
            name='numSets',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
    ]