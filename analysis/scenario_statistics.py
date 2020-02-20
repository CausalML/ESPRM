from analysis.evaluate_policy import evaluate_policy_cf_data


def get_scenario_statistics(scenario, policy_network_class, x_test, y_test_cf):
    optimal_policy = scenario.get_optimal_policy(
        policy_network_class, x_test, y_test_cf)
    optimal_policy_val = evaluate_policy_cf_data(
        optimal_policy, x_test, y_test_cf)
    optimal_theta = optimal_policy.get_policy_weights()
    worst_policy = optimal_policy.get_negative()
    worst_policy_val = evaluate_policy_cf_data(
        worst_policy, x_test, y_test_cf)
    return {
        "optimal_policy_val": optimal_policy_val,
        "worst_policy_val": worst_policy_val,
        "optimal_theta": optimal_theta,
    }
