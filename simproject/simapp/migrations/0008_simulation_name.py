# Generated by Django 5.0.4 on 2024-10-07 10:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('simapp', '0007_rowdetail_numsets'),
    ]

    operations = [
        migrations.AddField(
            model_name='simulation',
            name='name',
            field=models.CharField(default=0, max_length=50),
            preserve_default=False,
        ),
    ]
