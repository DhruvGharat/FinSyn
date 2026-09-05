import unittest
import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath("."))

from tests.test_reconciliation import (
    test_date_parsing,
    test_amount_normalization,
    test_candidate_generation,
    test_cost_calculation,
    test_hungarian_matching,
    test_unequal_row_counts,
    test_feature_generation,
    test_xgboost_inference,
    test_ground_truth_evaluation,
    test_exception_classification,
    test_audit_hash_generation,
    test_audit_chain_verification,
    test_tamper_detection,
    test_csv_validation,
    test_api_health,
    test_api_reconciliation,
    test_full_pipeline_end_to_end
)


class TestReconciliationSuite(unittest.TestCase):
    def test_01_date_parsing(self):
        test_date_parsing()

    def test_02_amount_normalization(self):
        test_amount_normalization()

    def test_03_candidate_generation(self):
        test_candidate_generation()

    def test_04_cost_calculation(self):
        test_cost_calculation()

    def test_05_hungarian_matching(self):
        test_hungarian_matching()

    def test_06_unequal_row_counts(self):
        test_unequal_row_counts()

    def test_07_feature_generation(self):
        test_feature_generation()

    def test_08_xgboost_inference(self):
        test_xgboost_inference()

    def test_09_ground_truth_evaluation(self):
        test_ground_truth_evaluation()

    def test_10_exception_classification(self):
        test_exception_classification()

    def test_11_audit_hash_generation(self):
        test_audit_hash_generation()

    def test_12_audit_chain_verification(self):
        test_audit_chain_verification()

    def test_13_tamper_detection(self):
        test_tamper_detection()

    def test_14_csv_validation(self):
        test_csv_validation()

    def test_15_api_health(self):
        test_api_health()

    def test_16_api_reconciliation(self):
        test_api_reconciliation()

    def test_17_full_pipeline_end_to_end(self):
        test_full_pipeline_end_to_end()


if __name__ == "__main__":
    unittest.main(verbosity=2)
