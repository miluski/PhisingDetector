from importlib import import_module

RESOURCE_MODULES = (
    'app.resources.root_redirect_resource',
    'app.resources.legacy_docs_resource',
    'app.resources.health_resource',
    'app.resources.extract_features_resource',
    'app.resources.analyze_email_resource',
    'app.resources.analyze_sms_resource'
)


def register_resources():
    for module in RESOURCE_MODULES:
        import_module(module)
