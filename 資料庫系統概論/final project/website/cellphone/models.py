# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models

class Data(models.Model):
    cellphone_id = models.IntegerField(primary_key=True)
    brand = models.CharField(max_length=100, blank=True, null=True)
    model = models.CharField(max_length=100, blank=True, null=True)
    internal_memory = models.FloatField(blank=True, null=True)
    ram = models.FloatField(blank=True, null=True)
    performance = models.FloatField(blank=True, null=True)
    main_camera = models.FloatField(blank=True, null=True)
    selfie_camera = models.FloatField(blank=True, null=True)
    battery_size = models.FloatField(blank=True, null=True)
    screen_size = models.FloatField(blank=True, null=True)
    weight = models.FloatField(blank=True, null=True)
    price = models.FloatField(blank=True, null=True)
    release_date = models.DateField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'data'

    class Meta:
        managed = False
        db_table = 'data'

class Rate(models.Model):
    user = models.OneToOneField('Users', models.DO_NOTHING, primary_key=True)
    cellphone = models.ForeignKey(Data, models.DO_NOTHING)
    rating = models.FloatField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'rate'
        unique_together = (('user', 'cellphone'),)


class Users(models.Model):
    user_id = models.CharField(primary_key=True, max_length=100)
    age = models.FloatField(blank=True, null=True)
    gender = models.CharField(max_length=100, blank=True, null=True)
    occupation = models.CharField(max_length=100, blank=True, null=True)
    password = models.CharField(max_length=100, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'users'
