
from configobj import ConfigObj, flatten_errors
from validate import Validator


class ConfigReader(object):
    '''
        This class loads and validates configuration
        given configuration and specification files.
        It is reserved for other custom actions but
        for now configuration can be accessed as
        `reader_obj.conf` once it is successfully loaded
    '''
    def __init__(self, conf_file, conf_specfile):

        if conf_file is None:
            self._conf = ConfigObj()
        else:
            config = ConfigObj(conf_file, configspec=conf_specfile,
                               file_error=True)
            validator = Validator()
            res = config.validate(validator, preserve_errors=True)

            # res is not boolean necessarily
            if res is True:
                self._conf = config
            else:
                self._conf = None
                self.print_validation_errors(config, res)
                raise ValueError('Failed to validate file {} using'
                                 ' specification {}'.format(conf_file,
                                                            conf_specfile))

    @classmethod
    def print_validation_errors(cls, config, res):
        for entry in flatten_errors(config, res):
            # each entry is a tuple
            section_list, key, error = entry
            if key is not None:
                section_list.append(key)
            else:
                section_list.append('[missing section]')
            section_string = ', '.join(section_list)
            if error is False:
                error = 'Missing value or section.'
            print section_string, ' = ', error

    @property
    def conf(self):
        return self._conf

