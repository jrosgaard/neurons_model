EXT files define extensions.
The functions in this folder are to allow for importing extension packages effectively.

When Making a new extension, ensure you have:
__init__.py and plugin.py

plugin.py should have the register function below:

def register(registry):
    """Register GBMsim with the neurons_model extension registry."""
    return registry.register_extension(
        package=__package__ or "_EXT_[PackageName]",
        name=_EXTENSION_NAME,
        version=_EXTENSION_VERSION,
        description=_EXTENSION_DESCRIPTION,
        author=_EXTENSION_AUTHOR,
        functions=_EXTENSION_FUNCTIONS,
        variables=_EXTENSION_VARIABLES,
    )

Although the actual entries may vary to mathc hte functionality of the given entension package.

