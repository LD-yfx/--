import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('reporting',Path(__file__).parents[1]/'quality_navigation_demo'/'reporting.py')
reporting=importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporting)

class ReportCompletionTests(unittest.TestCase):
    def setUp(self):
        self.cases=[{'case':name} for name in reporting.FULL_CASES]
        self.checks={key:True for key in reporting.STATIC_CHECKS+reporting.DYNAMIC_CHECKS+(reporting.INSTALLED_CHECK,)}

    def test_live_report_cannot_pass_even_with_every_result(self):
        result=reporting.completion_status('full',self.cases,self.checks,False)
        self.assertFalse(result['completed']);self.assertFalse(result['all_passed'])

    def test_static_results_cannot_claim_full_acceptance(self):
        result=reporting.completion_status('full',self.cases[:2],self.checks,True)
        self.assertFalse(result['completed']);self.assertFalse(result['all_passed'])

    def test_full_requires_all_eleven_checks(self):
        for removed in self.checks:
            subset={key:value for key,value in self.checks.items() if key!=removed}
            result=reporting.completion_status('full',self.cases,subset,True)
            self.assertFalse(result['completed']);self.assertEqual(result['missing_checks'],[removed])

    def test_duplicates_wrong_order_and_unexecuted_case_rejected(self):
        for cases in [self.cases[:-1],self.cases[:-1]+[self.cases[0]],list(reversed(self.cases))]:
            self.assertFalse(reporting.completion_status('full',cases,self.checks,True)['completed'])

    def test_completed_failure_is_not_a_pass(self):
        self.checks['dynamic_no_collisions']=False
        result=reporting.completion_status('full',self.cases,self.checks,True)
        self.assertTrue(result['completed']);self.assertFalse(result['all_passed'])

    def test_exception_and_truthy_non_boolean_never_pass(self):
        self.assertFalse(reporting.completion_status('full',self.cases,self.checks,True,error='exception')['all_passed'])
        self.checks['static_goal_reached']='true'
        self.assertFalse(reporting.completion_status('full',self.cases,self.checks,True)['all_passed'])

    def test_static_requires_four_or_five_checks_as_requested(self):
        checks={key:True for key in reporting.STATIC_CHECKS}
        self.assertTrue(reporting.completion_status('static_only',self.cases[:2],checks,True)['all_passed'])
        self.assertFalse(reporting.completion_status('static_only',self.cases[:2],checks,True,True)['completed'])
        checks[reporting.INSTALLED_CHECK]=True
        self.assertTrue(reporting.completion_status('static_only',self.cases[:2],checks,True,True)['all_passed'])

    def test_full_complete_case(self):
        result=reporting.completion_status('full',self.cases,self.checks,True)
        self.assertTrue(result['completed']);self.assertTrue(result['all_passed'])
        self.assertEqual(len(result['required_checks']),11)

if __name__=='__main__':unittest.main()
