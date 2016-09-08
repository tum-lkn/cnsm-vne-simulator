import errors.homebrewederrors as errorbrewer
import literals


class DataInterfaceConfig(object):
    def __init__(self, storage_type=None):
        self.storage_type = storage_type
        self.dict_writer = None

    def describe(self):
        # TODO Test me!
        """
        Describe the datainterface.

        Returns:

        """
        attributes = vars(self)

        return ', '.join("%s: %s" % item for item in attributes.items())

    def as_dict(self):
        return {
            'storage_type': self.storage_type
        }


class DatabaseDictConfigWriter():
    def as_dict(self, databaseconfig):
        return {
            'type': databaseconfig.storage_type,
            'database': databaseconfig.database,
            'host': databaseconfig.host,
            'port': databaseconfig.port,
            'user': databaseconfig.user,
            'passwd': databaseconfig.passwd
        }


class MySqlInterfaceConfig(DataInterfaceConfig):
    def __init__(self, database=None, host=None, port=None, user=None, passwd=None):
        super(MySqlInterfaceConfig, self).__init__(storage_type='mysql')
        self.database = database
        self.host = host
        self.port = port
        self.user = user
        self.passwd = passwd

    def as_dict(self):
        params = super(MySqlInterfaceConfig, self).as_dict()
        params['database'] = self.database
        params['host'] = self.host
        params['port'] = self.port
        params['user'] = self.user
        params['passwd'] = self.passwd
        return params


class PeeweeInterfaceConfig(DataInterfaceConfig):
    def __init__(self,
                 database=None,
                 host=None,
                 port=None,
                 user=None,
                 passwd=None):
        super(PeeweeInterfaceConfig, self).__init__(storage_type='mysql-peewee')
        self.database = database
        self.host = host
        self.port = port
        self.user = user
        self.passwd = passwd

    def as_dict(self):
        params = super(PeeweeInterfaceConfig, self).as_dict()
        params['database'] = self.database
        params['host'] = self.host
        params['port'] = self.port
        params['user'] = self.user
        params['passwd'] = self.passwd
        return params


class PymysqlInterfaceConfig(DataInterfaceConfig):
    def __init__(self,
                 database=None,
                 host=None,
                 port=None,
                 user=None,
                 passwd=None):
        super(PymysqlInterfaceConfig, self).__init__(storage_type='mysql-pymysql')
        self.database = database
        self.host = host
        self.port = port
        self.user = user
        self.passwd = passwd

    def as_dict(self):
        params = super(PymysqlInterfaceConfig, self).as_dict()
        params['database'] = self.database
        params['host'] = self.host
        params['port'] = self.port
        params['user'] = self.user
        params['passwd'] = self.passwd
        return params


class FileInterfaceConfig(DataInterfaceConfig):
    def __init__(self, destination=None):
        super(FileInterfaceConfig, self).__init__(storage_type='file')
        self.destination = destination

    def as_dict(self):
        params = super(FileInterfaceConfig, self).as_dict()
        params['destination'] = self.destination
        return params


class InterfaceConfigFactory(object):
    @classmethod
    def produce(cls, parameter):
        if parameter['storage_type'] == literals.STORAGE_MYSQL:
            return MySqlInterfaceConfig(database=parameter['database'],
                                        host=parameter['host'],
                                        port=int(parameter['port']),
                                        user=parameter['user'],
                                        passwd=parameter['passwd'])
        elif parameter['storage_type'] == literals.STORAGE_PEEWEE:
            return PeeweeInterfaceConfig(database=parameter['database'],
                                         host=parameter['host'],
                                         port=int(parameter['port']),
                                         user=parameter['user'],
                                         passwd=parameter['passwd'])
        elif parameter['storage_type'] == literals.STORAGE_PYMYSQL:
            return PymysqlInterfaceConfig(database=parameter['database'],
                                          host=parameter['host'],
                                          port=int(parameter['port']),
                                          user=parameter['user'],
                                          passwd=parameter['passwd'])
        elif parameter['storage_type'] is literals.STORAGE_FILE:
            return FileInterfaceConfig(destination=parameter['destination'])
        else:
            raise errorbrewer.InterfaceNotKnownError('Interface {} unknown'.format(parameter['type']))
