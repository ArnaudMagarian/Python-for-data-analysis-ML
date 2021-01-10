from django.db import models
from django.core.validators import MaxValueValidator, MinValueValidator

class DataSignification(models.Model):

    Variable = models.CharField(max_length=400)
    Signification = models.TextField()

    def __str__(self):
        #Show the name of the article in the admin page rather that object num
        return self.variable


GENDER_CHOICES =(
    ('Female', 'Female'),
    ('Male', 'Male')
)

YES_NO_CHOICES = (
    ('yes', 'yes'),
    ('no', 'no')
)

CAEC_CHOICES = (
    ('no', 'no'),
    ('Sometimes', 'Sometimes'),
    ('Frequently', 'Frequently'),
    ('Always', 'Always')
)

CALC_CHOICES = (
    ('no', 'no'),
    ('Sometimes', 'Sometimes'),
    ('Frequently', 'Frequently'),
    ('Always', 'Always')
)

MTRANS_CHOICES = (
    ('Automobile', 'Automobile'),
    ('Public_Transportation', 'Public_Transportation'),
    ('Walking', 'Walking')
)

NOBEYESDAD_CHOICES = (
    ('Insufficient_Weight', 'Insufficient_Weight'),
    ('Normal_Weight', 'Normal_Weight'),
    ('Overwieght_Level_I', 'Overwieght_Level_I'),
    ('Overwieght_Level_II', 'Overwieght_Level_II'),
    ('Obesity_Type_I', 'Obesity_Type_I'),
    ('Obesity_Type_II', 'Obesity_Type_II'),
    ('Obesity_Type_III', 'Obesity_Type_III')
)


class dataset(models.Model):
    Gender = models.CharField(max_length=6, choices=GENDER_CHOICES)
    Age = models.PositiveIntegerField(validators=[MinValueValidator(0), MaxValueValidator(150)])
    Height = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(3.0)])
    Weight = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(400.0)])
    family_history_with_overweight = models.CharField(max_length=3, choices=YES_NO_CHOICES)
    FAVC = models.CharField(max_length=3, choices=YES_NO_CHOICES)
    FCVC = models.PositiveIntegerField(validators=[MinValueValidator(0), MaxValueValidator(10)])
    NCP = models.PositiveIntegerField(validators=[MinValueValidator(0), MaxValueValidator(10)])
    CAEC = models.CharField(max_length=10, choices=CAEC_CHOICES)
    SMOKE = models.CharField(max_length=3, choices=YES_NO_CHOICES)
    CH2O = models.PositiveIntegerField(validators=[MinValueValidator(0), MaxValueValidator(10)])
    SCC = models.CharField(max_length=3, choices=YES_NO_CHOICES)
    FAF = models.PositiveIntegerField(validators=[MinValueValidator(0), MaxValueValidator(10)])
    TUE = models.PositiveIntegerField(validators=[MinValueValidator(0), MaxValueValidator(10)])
    CALC = models.CharField(max_length=10, choices=CALC_CHOICES)
    MTRANS = models.CharField(max_length=21, choices=MTRANS_CHOICES)
    NObeyesdad = models.CharField(max_length=19, choices=NOBEYESDAD_CHOICES)
