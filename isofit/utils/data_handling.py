#! /usr/bin/env python3
#
#  Copyright 2019 California Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# ISOFIT: Imaging Spectrometer Optimal FITting
# Author: Niklas Bohn, urs.n.bohn@jpl.nasa.gov
#

class DataHandling(object):
    """

    """
    def __init__(self, file_dir):
        """

        """
        self.input_dile_directory = file_dir

    def get_tutorial_data(self, type):
        """

        """
        if type == "EMIT_snow":
            url = 'https://drive.google.com/uc?export=download&id=1-__e8dMlJMC_Lbt5q9-LwS-L5wghKYnn'
            gdown.download(url, self.input_dile_directory, quiet=False)

    def prepare_input_data(self):
        """

        """
