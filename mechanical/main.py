import unittest
import tests.unittests as test

suite = unittest.defaultTestLoader.loadTestsFromModule(test)
unittest.TextTestRunner(verbosity=2).run(suite)