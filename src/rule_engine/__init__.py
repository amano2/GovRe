from .hr_rules import evaluate_hr_rule
from .finance_rules import evaluate_finance_rule
from .it_access_rules import evaluate_it_access_rule

RULE_EVALUATORS = {
    "hr": evaluate_hr_rule,
    "finance": evaluate_finance_rule,
    "it_access": evaluate_it_access_rule,
}

__all__ = [
    "evaluate_hr_rule",
    "evaluate_finance_rule",
    "evaluate_it_access_rule",
    "RULE_EVALUATORS",
]
