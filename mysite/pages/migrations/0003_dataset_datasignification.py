# Generated by Django 3.0.5 on 2021-01-10 19:41

import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('pages', '0002_auto_20210110_2032'),
    ]

    operations = [
        migrations.CreateModel(
            name='dataset',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Gender', models.CharField(choices=[('Female', 'Female'), ('Male', 'Male')], max_length=6)),
                ('Age', models.PositiveIntegerField(validators=[django.core.validators.MinValueValidator(0), django.core.validators.MaxValueValidator(150)])),
                ('Height', models.FloatField(validators=[django.core.validators.MinValueValidator(0.0), django.core.validators.MaxValueValidator(3.0)])),
                ('Weight', models.FloatField(validators=[django.core.validators.MinValueValidator(0.0), django.core.validators.MaxValueValidator(400.0)])),
                ('family_history_with_overweight', models.CharField(choices=[('yes', 'yes'), ('no', 'no')], max_length=3)),
                ('FAVC', models.CharField(choices=[('yes', 'yes'), ('no', 'no')], max_length=3)),
                ('FCVC', models.PositiveIntegerField(validators=[django.core.validators.MinValueValidator(0), django.core.validators.MaxValueValidator(10)])),
                ('NCP', models.PositiveIntegerField(validators=[django.core.validators.MinValueValidator(0), django.core.validators.MaxValueValidator(10)])),
                ('CAEC', models.CharField(choices=[('no', 'no'), ('Sometimes', 'Sometimes'), ('Frequently', 'Frequently'), ('Always', 'Always')], max_length=10)),
                ('SMOKE', models.CharField(choices=[('yes', 'yes'), ('no', 'no')], max_length=3)),
                ('CH2O', models.PositiveIntegerField(validators=[django.core.validators.MinValueValidator(0), django.core.validators.MaxValueValidator(10)])),
                ('SCC', models.CharField(choices=[('yes', 'yes'), ('no', 'no')], max_length=3)),
                ('FAF', models.PositiveIntegerField(validators=[django.core.validators.MinValueValidator(0), django.core.validators.MaxValueValidator(10)])),
                ('TUE', models.PositiveIntegerField(validators=[django.core.validators.MinValueValidator(0), django.core.validators.MaxValueValidator(10)])),
                ('CALC', models.CharField(choices=[('no', 'no'), ('Sometimes', 'Sometimes'), ('Frequently', 'Frequently'), ('Always', 'Always')], max_length=10)),
                ('MTRANS', models.CharField(choices=[('Automobile', 'Automobile'), ('Public_Transportation', 'Public_Transportation'), ('Walking', 'Walking')], max_length=21)),
                ('NObeyesdad', models.CharField(choices=[('Insufficient_Weight', 'Insufficient_Weight'), ('Normal_Weight', 'Normal_Weight'), ('Overwieght_Level_I', 'Overwieght_Level_I'), ('Overwieght_Level_II', 'Overwieght_Level_II'), ('Obesity_Type_I', 'Obesity_Type_I'), ('Obesity_Type_II', 'Obesity_Type_II'), ('Obesity_Type_III', 'Obesity_Type_III')], max_length=19)),
            ],
        ),
        migrations.CreateModel(
            name='DataSignification',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Variable', models.CharField(max_length=400)),
                ('Signification', models.TextField()),
            ],
        ),
    ]