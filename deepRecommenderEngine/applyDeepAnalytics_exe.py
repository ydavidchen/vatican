__author__ = "David Chen"
__credit__ = ["David Chen","Won Murdocq","Derrick Wilson-Duncan"]
__copyright__ = "Copyright 2018"
__license__ = "GPL3.0"
__version__ = "1.0.0"
__status__ = "Production"

from trainDeepLearnCV import *
from applyDeepAnalytics import *

if __name__ == '__main__':
	testCase = loadAndApply(); 
	visualizeAndExport(testCase, '../public/deepLearnApp/test_case/recommendation.png')

quit();