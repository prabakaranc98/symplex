"""Model output contracts shared by agents and interfaces."""

from symplex.core.contracts import TEXT, TEXTS, Invalid, closed, schema


class Review:
    @staticmethod
    def json_schema():
        return schema(
            {
                "verdict": {
                    "type": "string",
                    "enum": [
                        "supported_within_scope",
                        "inconclusive",
                        "contradicted_under_test",
                        "invalid_test",
                    ],
                },
                "result_ids": TEXTS,
                "concerns": TEXTS,
                "next_check": TEXT,
            }
        )

    @staticmethod
    def parse(value):
        closed(value, ("verdict", "result_ids", "concerns", "next_check"))
        if value["verdict"] not in (
            "supported_within_scope",
            "inconclusive",
            "contradicted_under_test",
            "invalid_test",
        ):
            raise Invalid("Unknown review verdict")
        return value


class CodeProposal:
    @staticmethod
    def json_schema():
        return schema(
            {
                "filename": TEXT,
                "code": TEXT,
                "expectation": TEXT,
                "required_checks": TEXTS,
                "dependencies": TEXTS,
            }
        )

    @staticmethod
    def parse(value):
        closed(
            value,
            ("filename", "code", "expectation", "required_checks", "dependencies"),
        )
        if not isinstance(value["code"], str) or len(value["code"]) > 30000:
            raise Invalid("Code proposal exceeds permitted size")
        return value


class DeliveryReview:
    @staticmethod
    def json_schema():
        return schema(
            {
                "assessment": TEXT,
                "unsupported_claims": {"type": "array", "items": TEXT},
                "next_validation": TEXT,
            }
        )

    @staticmethod
    def parse(data):
        from symplex.core.contracts import closed

        closed(data, ("assessment", "unsupported_claims", "next_validation"))
        return data
