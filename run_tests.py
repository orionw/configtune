import sys
import unittest

test_runner = unittest.TextTestRunner(verbosity=1)
tests = unittest.TestLoader().discover('test')

if not test_runner.run(tests).wasSuccessful():
    sys.exit(1)
