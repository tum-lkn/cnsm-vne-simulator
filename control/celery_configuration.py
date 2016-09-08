import kombu

"""
Contains the configuration of Celery like the Broker or the routing of tasks that should apply
"""


class CeleryRouter(object):
    """
    Modifies the routing of the tasks according to the problem type.
    """

    def route_for_task(self, task, args=None, kwargs=None):
        try:
            problem_type = args[0]['problem_type']
            if problem_type == 'dhpp':
                return {'queue': 'dhpp', 'routing_key': 'dhpp'}
            elif problem_type == 'cpp':
                return {'queue': 'cpp', 'routing_key': 'cpp'}
            elif problem_type == 'vne':
                return {'queue': 'vne', 'routing_key': 'vne'}
        except KeyError:
            pass
        except TypeError:
            pass
        return None


CELERY_RESULT_BACKEND = 'rpc://'
BROKER_URL = 'amqp://dhpp:1q2w3e4r@10.152.13.7:5672/dhpp_testing'

CELERY_QUEUES = (
    kombu.Queue('dhpp', kombu.Exchange('default', type='direct'), routing_key='dhpp'),
    kombu.Queue('vne', kombu.Exchange('default', type='direct'), routing_key='vne'),
    kombu.Queue('cpp', kombu.Exchange('default', type='direct'), routing_key='cpp')
)
CELERY_DEFAULT_QUEUE = 'dhpp'
CELERY_DEFAULT_EXCHANGE = 'default'
CELERY_DEFAULT_ROUTING_KEY = 'dhpp'
CELERY_ROUTES = (CeleryRouter(),)
