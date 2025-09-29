from __future__ import annotations

import functools
import itertools
import sys
from concurrent.futures import Executor
from concurrent.futures import Future
from types import TracebackType
from typing import Any
from typing import Callable
from typing import Generator
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import TypeVar

if sys.version_info >= (3, 10):  # pragma: >=3.10 cover
    from typing import ParamSpec
else:  # pragma: <3.10 cover
    from typing_extensions import ParamSpec

if sys.version_info >= (3, 11):  # pragma: >=3.11 cover
    from typing import Self
else:  # pragma: <3.11 cover
    from typing_extensions import Self

P = ParamSpec('P')
T = TypeVar('T')


def _get_chunks(
    *iterables: Iterable[T],
    chunksize: int,
) -> Generator[tuple[tuple[T, ...], ...], None, None]:
    it = zip(*iterables)
    while True:
        chunk = tuple(itertools.islice(it, chunksize))
        if not chunk:
            return
        yield chunk


def _process_chunk(
    function: Callable[P, T],
    chunk: Iterable[P.args],
) -> list[T]:
    return [function(*args) for args in chunk]


class _Task(Generic[P, T]):
    def __init__(
        self,
        executor: Executor,
        function: Callable[P, T],
        args: P.args,
        kwargs: P.kwargs,
        client_future: Future[T],
    ) -> None:
        self.executor = executor
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.client_future = client_future
        self.task_future: Future[T] | None = None
        self.pending_futures: set[Future[Any]] = set()

        for arg in [*args, *kwargs.values()]:
            if isinstance(arg, Future) and not arg.done():
                arg.add_done_callback(self._pending_future_callback)
                self.pending_futures.add(arg)

        if len(self.pending_futures) == 0:
            self._submit()

    def _pending_future_callback(self, future: Future[Any]) -> None:
        assert future.done()
        assert len(self.pending_futures) > 0
        if future in self.pending_futures:
            self.pending_futures.remove(future)
        else:  # pragma: no cover
            return

        if future.cancelled():
            self.client_future.cancel()
        elif future.exception() is not None:
            self.client_future.set_exception(future.exception())
        elif len(self.pending_futures) == 0:
            self._submit()

    def _task_future_callback(self, future: Future[T]) -> None:
        assert future.done()
        if future.cancelled():
            self.client_future.cancel()
        elif future.exception() is not None:
            self.client_future.set_exception(future.exception())
        else:
            self.client_future.set_result(future.result())

    def _submit(self) -> None:
        assert self.task_future is None

        if not self.client_future.set_running_or_notify_cancel():
            # client_future was cancelled so don't submit the task.
            return

        args = tuple(
            arg.result() if isinstance(arg, Future) else arg
            for arg in self.args
        )
        kwargs = {
            key: value.result() if isinstance(value, Future) else value
            for key, value in self.kwargs.items()
        }

        func_name = self.function.function.__name__
        func_uuids = {
            "potrf": "fb5c2e2b-a0a4-4198-8d12-f3db86c6f5c5",
            "trsm": "426cde7b-0c35-4aa6-ad52-66272ba91c4d",
            "syrk": "88df2d12-07d7-4fc7-b73b-47f25bc004d7",
            "gemm": "b7ac0fc0-110c-4df0-b456-58d88f2bd0bf",
            "smi_to_pdb": "5acf28a9-9895-422f-a4b8-8659665e0a64",
            "set_element": "afa80ad7-70a7-4dce-b16f-6a99cf4cc6f3",
            "pdb_to_pdbqt": "7e37747f-1e60-4054-9128-dd944526b94c",
            "make_autodock_config": "b53f716c-0eca-472f-8915-6020d40d6a95",
            "autodock_vina": "257bfb80-570b-452b-80b3-400ed26a074a",
            "test_model": "8a838a77-9d0c-4237-98cf-2cbdb06f206d",
            "no_local_train": "464429a8-8373-4f7c-a2b2-5327df921c51",
            "local_train": "1c6309db-e7bd-4f38-a2e5-29509457a3f7",
            "map_task": "46468746-c650-4a02-b88a-5b706a25cfa6",
            "reduce_task": "7c3f35ff-2a74-4b3f-a09e-ff6126709f63",
            "compute_vertical": "f4a7872e-aa8b-4701-b26a-c4c00681583a",
            "train_model": "0400a428-4fde-400b-8bb0-b4bf039cf1f9",
            "run_model": "edebd843-66a2-4489-ac21-c097b86f9134",
            "combine_inferences": "f01c3561-3f4e-4e0e-a37f-228b0ce917f4",
            "configure_montage": "dac54ac2-1205-4f17-b63f-afc74bf38aba",
            "mproject": "fec81aef-b3f7-4a6c-b3a9-958b273ddcd3",
            "mimgtbl": "2c15f325-8638-4ec1-ab36-cadd4f792d18",
            "moverlaps": "a2d507b4-4230-423c-86e1-157ba57a66b6",
            "mdiff": "4931f62f-7916-4968-908b-2f050672e2a0",
            "bgexec_prep": "12d141c4-d06c-46b7-a28e-bd485a69902b",
            "mbackground": "7ff41218-7240-4d68-a025-5a623286c9a8",
            "madd": "47c845f8-050e-469d-a9c2-3ef5a801be97",
            "read_csv": "11bef9e4-26ff-40e2-815c-de6e0eb8519a",
            "noop_task": "ffac5336-75fa-4202-a58b-0a76c0829619",
            "warmup_task": "39ce2c88-1b28-4f45-ab3d-05e6bd664540"
        }
        
        print(func_name)
        #self.task_future = self.executor.submit(self.function, *args, **kwargs)
        self.task_future = self.executor.submit_to_registered_function(func_uuids[func_name], args, kwargs)
        self.task_future.add_done_callback(self._task_future_callback)


class FutureDependencyExecutor(Executor):
    """Executor wrapper that adds DAG-like features.

    An [`Executor`][concurrent.futures.Executor] implementation that wraps
    another executor with logic for delaying task submission until all
    [`Future`][concurrent.futures.Future] instances which are args or kwargs
    of a task have completed. In other words, child tasks will not be
    scheduled until the results of the child's parent tasks are available.

    Args:
        executor: Executor to wrap.
    """

    def __init__(self, executor: Executor) -> None:
        self.executor = executor
        self._tasks: dict[Future[Any], _Task[Any, Any]] = {}

    def __enter__(self) -> Self:
        self.executor.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ) -> bool | None:
        return self.executor.__exit__(exc_type, exc_value, exc_traceback)

    def _task_future_callback(self, future: Future[Any]) -> None:
        self._tasks.pop(future)

    def submit(
        self,
        function: Callable[P, T],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Future[T]:
        """Schedule the callable to be executed.

        Args:
            function: Callable to execute.
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            [`Future`][concurrent.futures.Future] object representing the \
            result of the execution of the callable.
        """
        client_future: Future[T] = Future()
        task = _Task(self.executor, function, args, kwargs, client_future)
        self._tasks[client_future] = task
        client_future.add_done_callback(self._task_future_callback)
        return client_future

    def map(
        self,
        function: Callable[P, T],
        *iterables: Iterable[P.args],
        timeout: float | None = None,
        chunksize: int = 1,
    ) -> Iterator[T]:
        """Map a function onto iterables of arguments.

        Args:
            function: A callable that will take as many arguments as there are
                passed iterables.
            iterables: Variable number of iterables.
            timeout: The maximum number of seconds to wait. If None, then there
                is no limit on the wait time.
            chunksize: If greater than one, the iterables will be chopped into
                chunks of size chunksize and submitted to the executor. If set
                to one, the items in the list will be sent one at a time.

        Returns:
            An iterator equivalent to: `map(func, *iterables)` but the calls \
            may be evaluated out-of-order.

        Raises:
            ValueError: if chunksize is less than one.
        """
        # Based on concurrent.futures.ProcessPoolExecutor.map()
        # https://github.com/python/cpython/blob/37959e25cbbe1d207c660b5bc9583b9bd1403f1a/Lib/concurrent/futures/process.py
        if chunksize < 1:
            raise ValueError('chunksize must be >= 1.')

        results = super().map(
            functools.partial(_process_chunk, function),
            _get_chunks(*iterables, chunksize=chunksize),
            timeout=timeout,
        )

        def _result_iterator(
            iterable: Iterator[list[T]],
        ) -> Generator[T, None, None]:
            for element in iterable:
                element.reverse()
                while element:
                    yield element.pop()

        return _result_iterator(results)

    def shutdown(
        self,
        wait: bool = True,
        *,
        cancel_futures: bool = False,
    ) -> None:
        """Shutdown the executor.

        Args:
            wait: Wait on all pending futures to complete.
            cancel_futures: Cancel all pending futures that the executor
                has not started running. Only used in Python 3.9 and later.
        """
        if sys.version_info >= (3, 9):  # pragma: >=3.9 cover
            self.executor.shutdown(wait=wait, cancel_futures=cancel_futures)
        else:  # pragma: <3.9 cover
            self.executor.shutdown(wait=wait)
