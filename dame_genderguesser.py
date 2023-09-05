#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (C) 2020  David Arroyo Menéndez (davidam@gmail.com)
# This file is part of Damegender.

# Author: David Arroyo Menéndez <davidam@gmail.com>
# Maintainer: David Arroyo Menéndez <davidam@gmail.com>

# This file is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.

# This file is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with DameGender; see the file GPL.txt.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
# Boston, MA 02110-1301 USA,


import csv
import requests
import json
import os.path
import codecs
from app.dame_gender import Gender
from app.dame_utils import DameUtils
du = DameUtils()


class DameGenderGuesser(Gender):
    COUNTRIES = """
       great_britain ireland usa italy malta portugal spain france
       belgium luxembourg the_netherlands east_frisia germany austria
       swiss iceland denmark norway sweden finland estonia latvia
       lithuania poland czech_republic slovakia hungary romania
       bulgaria bosniaand croatia kosovo macedonia montenegro serbia
       slovenia albania greece russia belarus moldova ukraine armenia
       azerbaijan georgia the_stans turkey arabia israel china india
       japan korea vietnam other_countries
       """.split()

    def __init__(self,
                 case_sensitive=True):

        """Creates a detector parsing given data file"""
        self.case_sensitive = case_sensitive
        self._parse("files/names/nam_dict.txt")
        # Dictionary of countries and person names in files/names/nam_dict.txt
        self.keyscountries = {
            "30": "Great Britain", "31": "Ireland", "32": "USA",
            "33": "Italy", "34": "Malta", "35": "Portugal",
            "36": "Spain", "37": "France", "38": "Belgium",
            "39": "Luxembourg", "40": "The Netherlands",
            "41": "East Frisia", "42": "Germany", "43": "Austria",
            "44": "Swiss", "45": "Iceland", "46": "Denmark",
            "47": "Norway", "48": "Sweden", "49": "Finland",
            "50": "Estonia", "51": "Latvia", "52": "Lithuania",
            "53": "Poland", "54": "Czech Republic", "55": "Slovakia",
            "56": "Hungary", "57": "Romania", "58": "Bulgaria",
            "59": "Bosnia and Herzegovina", "60": "Croatia",
            "61": "Kosovo", "62": "Macedonia", "63": "Montenegro",
            "64": "Serbia", "65": "Slovenia", "66": "Albania",
            "67": "Greece", "68": "Rusia", "69": "Belarus",
            "70": "Moldova", "71": "Ukraine", "72": "Armenia",
            "73": "Azerbaijan", "74": "Georgia",
            "75": "Kazakhstan/Uzbekistan", "76": "Turkey",
            "77": "Arabia/Persia", "78": "Israel", "79": "China",
            "80": "India/Sri Lanka", "81": "Japan", "82": "Korea",
            "83": "Vietnam", "84": "other countries"}

    def _parse(self, filename):
        # Open a data file and for each line, call the _eat_name_line function.
        """Opens data file and for each line, calls _eat_name_line"""
        self.names = {}
        with codecs.open(filename, encoding="utf-8") as f:
            for line in f:
                self._eat_name_line(line.strip())

    def _eat_name_line(self, line):
        # Used to parse a line of data from a file.
        # If the first letter of the line is not "#" or "=",
        # the code will split the line into parts and
        # extract the country values ​​from the line from
        # character 30 to the last. Then the code will get
        # the name of the second part and convert it to
        # lower case if it is not case sensitive.
        # Depending on the gender value (the first part)
        # found on the line, the code will set a value
        # for that name (eg "male", "female", etc.)
        # along with the corresponding country values.
        # If the gender value is not recognized,
        # it will throw an error.
        """Parses one line of data file"""
        if line[0] not in "#=":
            parts = line.split()
            country_values = line[30:-1]
            name = parts[1]
            if not self.case_sensitive:
                name = name.lower()
            if parts[0] == "M":
                self._set(name, u"male", country_values)
            elif parts[0] == "1M" or parts[0] == "?M":
                self._set(name, u"mostly_male", country_values)
            elif parts[0] == "F":
                self._set(name, u"female", country_values)
            elif parts[0] == "1F" or parts[0] == "?F":
                self._set(name, u"mostly_female", country_values)
            elif parts[0] == "?":
                self._set(name, u"andy", country_values)
            else:
                raise "Not sure what to do with a sex of %s" % parts[0]

    def _set(self, name, gender, country_values):
        # Sets gender and relevant country values
        # for names dictionary of detector
        if '+' in name:
            for replacement in ['', ' ', '-']:
                self._set(name.replace('+', replacement),
                          gender,
                          country_values)
        else:
            if name not in self.names:
                self.names[name] = {}
            self.names[name][gender] = country_values

    def _most_popular_gender(self, name, counter):
        # Finds the most popular gender for the
        # given name counting by given counter
        if name not in self.names:
            return u"unknown"

        max_count, max_tie = (0, 0)
        best = list(self.names[name].keys())[0]
        for gender, country_values in list(self.names[name].items()):
            count, tie = counter(country_values)
            if count > max_count or (count == max_count and tie > max_tie):
                max_count, max_tie, best = count, tie, gender

        return best if max_count > 0 else u"andy"

    def get_gender(self, name, country=None):
        # Returns best gender for the given name and country pair
        if not self.case_sensitive:
            name = name.lower()
        if name not in self.names:
            return u"unknown"
        elif not country:
            def counter(country_values):
                country_values = list(map(ord,
                                          country_values.replace(" ", "")))
                return (len(country_values),
                        sum([c > 64 and c-55 or c-48 for c in country_values]))
            return self._most_popular_gender(name, counter)
        elif country in self.__class__.COUNTRIES:
            index = self.__class__.COUNTRIES.index(country)
            counter = lambda e: (ord(e[index])-32, 0)
            return self._most_popular_gender(name, counter)
        else:
            raise NoCountryError("No such country: %s" % country)

    def guess(self, name, gender_encoded=False):
        # guess method to check names dictionary
        genderguesserlist = []
#        d = gender.Detector()
        get = self.get_gender(name)
        if (((get == 'female') or (get == 'mostly_female')) and gender_encoded):
            guess = 0
        elif (((get == 'male') or (get == 'mostly_male')) and gender_encoded):
            guess = 1
        elif (((get == 'unknown') or (get == 'andy')) and gender_encoded):
            guess = 2
        else:
            guess = get
        return guess

    def guess_list(self, path='files/names/partial.csv',
                   gender_encoded=False, *args, **kwargs):
        # guess list method
        header = kwargs.get('header', True)
        slist = []
        with open(path) as csvfile:
            sexreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            if header:
                next(sexreader, None)
            for row in sexreader:
                name = row[0].title()
                name = name.replace('\"', '')
                slist.append(self.guess(name, gender_encoded))
        return slist

    def exists_in_country(self, num, arr):
        # Checks if a given item exists in a given country.
        bool1 = (str(arr[num]) == "A") or (str(arr[num]) == "B")
        bool1 = bool1 or (str(arr[num]) == "C")
        bool1 = bool1 or (str(arr[num]) == "D")
        if du.is_not_blank(arr[num]):
            if bool1:
                return True
            elif (int(arr[num]) > 0):
                return True
            else:
                return False
        else:
            return False
