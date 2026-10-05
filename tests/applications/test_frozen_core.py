"""Local execution authorization checks; no Git mutation or numerical launch."""
import subprocess

import pytest

from tools.denise_case import fluid2


def fake_git(monkeypatch, *, head=fluid2.CORE_SHA, origin=fluid2.CORE_SHA,
             dirty='', untracked='', staged=''):
    calls=[]

    def read_only(repo,*args):
        calls.append(args)
        if args==('rev-parse','HEAD'):
            return head
        if args==('rev-parse','--verify','--quiet','refs/remotes/origin/modernization'):
            if origin is None:
                raise subprocess.CalledProcessError(1,['git',*args])
            return origin
        if args[0]=='diff' and '--cached' in args:
            return staged
        if args[0]=='diff':
            assert args[1]==fluid2.CORE_SHA
            assert args[-4:]==('src','include','tests/physics','tests/utilities')
            return dirty
        if args[0]=='ls-files':
            return untracked
        if args==('branch','--show-current'):
            return ''
        raise AssertionError(f'network/ref mutation or unexpected Git command: {args}')

    monkeypatch.setattr(fluid2,'git',read_only)
    return calls


@pytest.mark.parametrize('origin',[fluid2.CORE_SHA,'a'*40,None])
def test_exact_authorized_head_accepts_equal_ahead_or_unavailable_origin(monkeypatch,origin):
    calls=fake_git(monkeypatch,origin=origin)
    result=fluid2.require_core('.',authorized_sha=fluid2.CORE_SHA,remote=True)
    assert result['head']==result['authorized_core_sha']==fluid2.CORE_SHA
    assert result['observed_repository_state']['origin_modernization']==origin
    assert not result['live_remote_checked']
    assert all(call[0] in ('rev-parse','diff','ls-files','branch') for call in calls)


def test_wrong_local_head_rejected_before_origin_probe(monkeypatch):
    calls=fake_git(monkeypatch,head='b'*40)
    with pytest.raises(ValueError,match='local HEAD'):
        fluid2.require_core('.')
    assert calls==[('rev-parse','HEAD')]


def test_explicit_wrong_authorized_sha_rejected(monkeypatch):
    fake_git(monkeypatch)
    with pytest.raises(ValueError,match='authorized frozen'):
        fluid2.require_core('.',authorized_sha='c'*40)


def test_authorization_must_be_full_sha_not_head_or_branch(monkeypatch):
    calls=fake_git(monkeypatch)
    with pytest.raises(ValueError,match='full authorized'):
        fluid2.require_core('.',authorized_sha='HEAD')
    assert not calls


@pytest.mark.parametrize('change',[{'dirty':'src/PSV/operator.c'},
                                 {'untracked':'src/PSV/new_operator.c'},
                                 {'dirty':'tests/physics/oracle.py'},
                                 {'staged':'anything'}])
def test_protected_source_or_index_modification_rejected(monkeypatch,change):
    fake_git(monkeypatch,**change)
    with pytest.raises(ValueError,match='production/core tests or index'):
        fluid2.require_core('.')
