from copy import deepcopy
import pytest
from acdc.contract import ContractError
from acdc.validate_performance_comparison import paired_records


def records():
    first=[dict(phase='ED',center_index=i,reused=False,synthetic=False,seconds=.1,
                backend='reference_torch',input_sha256=str(i)*64,head_sha256='a'*64) for i in range(2)]
    second=deepcopy(first)
    for row in second:row['backend']='pl'
    return first,second


def test_same_output_does_not_hide_different_input():
    a,b=records();b[0]['input_sha256']='f'*64
    with pytest.raises(ContractError,match='identity'):paired_records(a,b)


def test_cached_or_missing_sample_cannot_improve_comparison():
    a,b=records();b[0]['reused']=True
    with pytest.raises(ContractError,match='reused'):paired_records(a,b)
    a,b=records()
    with pytest.raises(ContractError,match='sample sets'):paired_records(a,b[1:])


def test_exact_matched_samples_accept_only_physical_pl():
    a,b=records();assert paired_records(a,b)==2
    b[0]['backend']='demo'
    with pytest.raises(ContractError,match='backend'):paired_records(a,b)
