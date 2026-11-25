"""Unit tests for mobench/platform/cpu.py (add-cpu-pin-lockfreq 取核算法).

The selection rule must never *partially* occupy a cluster: threads placed on
part of a cluster run at a different frequency from their peers, which is the
noise this feature exists to remove.
"""
import pytest

from mobench.platform.cpu import (Policy, CpuSelectionError, cpu_mask,
                                  parse_policies, select_cpus)

# 1+3+4 three-cluster device (Snapdragon 865 / Xiaomi Mi 10):
# cpu0-3 little A55 1.80GHz / cpu4-6 big A77 2.42GHz / cpu7 prime 2.84GHz
SM8250 = [
    Policy(cpus=[0, 1, 2, 3], max_freq_khz=1804800),
    Policy(cpus=[4, 5, 6], max_freq_khz=2419200),
    Policy(cpus=[7], max_freq_khz=2841600),
]


def _cluster_of(cpu, policies):
    """Index of the policy owning ``cpu`` (for partial-occupancy assertions)."""
    for i, p in enumerate(policies):
        if cpu in p.cpus:
            return i
    raise AssertionError(f"cpu {cpu} not in any policy")


def _assert_no_partial_cluster(selected, policies):
    """Every cluster touched by ``selected`` must be touched entirely."""
    for i, p in enumerate(policies):
        taken = [c for c in p.cpus if c in selected]
        assert not taken or len(taken) == len(p.cpus), (
            f"cluster {p.cpus} partially occupied by {selected}")


# -- cluster-aware selection ------------------------------------------------


def test_single_thread_picks_highest_freq_core():
    # Act
    cpus = select_cpus(SM8250, 1)

    # Assert
    assert cpus == [7]


def test_two_threads_stay_within_one_cluster():
    """N=2: no whole-cluster combination exists -> front of the fastest
    cluster that can hold N, NOT prime + one big core."""
    # Act
    cpus = select_cpus(SM8250, 2)

    # Assert
    assert cpus == [4, 5]


def test_three_threads_hit_the_big_cluster_whole():
    """N=3 must be the whole big cluster, not {7,4,5} (partial big cluster)."""
    # Act
    cpus = select_cpus(SM8250, 3)

    # Assert
    assert cpus == [4, 5, 6]


def test_four_threads_take_prime_plus_big_cluster():
    """N=4: exact union of whole clusters {7} + {4,5,6} = cpu4-7."""
    # Act
    cpus = select_cpus(SM8250, 4)

    # Assert
    assert cpus == [4, 5, 6, 7]


def test_all_cores():
    # Act
    cpus = select_cpus(SM8250, 8)

    # Assert
    assert cpus == [0, 1, 2, 3, 4, 5, 6, 7]


def test_selection_never_partially_occupies_a_cluster():
    """Invariant for every N where a whole-cluster cover exists.

    N=2 and N=6 are excluded on purpose: no union of whole clusters sums to
    them (cluster sizes are 1/3/4), so a partial front is unavoidable.
    """
    # Act / Assert
    for n in (1, 3, 4, 5, 7, 8):
        selected = select_cpus(SM8250, n)
        assert len(selected) == n
        _assert_no_partial_cluster(selected, SM8250)


def test_no_whole_cluster_cover_falls_back_to_cluster_front():
    """N=2 / N=6: documented fallback, still confined to the fastest cluster
    whenever one can hold N."""
    # Act / Assert
    assert select_cpus(SM8250, 2) == [4, 5]
    assert select_cpus(SM8250, 6) == [0, 1, 4, 5, 6, 7]


def test_threads_beyond_core_count_returns_all_cores():
    """Oversubscription (threads > cores) binds to every core, no error."""
    # Act
    cpus = select_cpus(SM8250, 16)

    # Assert
    assert cpus == [0, 1, 2, 3, 4, 5, 6, 7]


def test_selection_ignores_policy_input_order():
    # Arrange
    shuffled = [SM8250[2], SM8250[0], SM8250[1]]

    # Act
    cpus = select_cpus(shuffled, 4)

    # Assert
    assert cpus == [4, 5, 6, 7]


# -- other device shapes ----------------------------------------------------


def test_single_homogeneous_cluster():
    # Arrange
    one = [Policy(cpus=[0, 1, 2, 3], max_freq_khz=2000000)]

    # Act / Assert
    assert select_cpus(one, 1) == [0]
    assert select_cpus(one, 2) == [0, 1]
    assert select_cpus(one, 4) == [0, 1, 2, 3]


def test_single_core_device():
    # Arrange
    one = [Policy(cpus=[0], max_freq_khz=2000000)]

    # Act / Assert
    assert select_cpus(one, 1) == [0]


def test_two_clusters_without_prime():
    # Arrange: 1 big + 3 little, big is cpu3
    two = [Policy(cpus=[0, 1, 2], max_freq_khz=2000000),
           Policy(cpus=[3], max_freq_khz=2500000)]

    # Act / Assert
    assert select_cpus(two, 1) == [3]
    assert select_cpus(two, 2) == [0, 1]
    assert select_cpus(two, 3) == [0, 1, 2]
    assert select_cpus(two, 4) == [0, 1, 2, 3]


def test_empty_policies_rejected():
    # Act / Assert
    with pytest.raises(CpuSelectionError, match="no CPUs"):
        select_cpus([], 1)


# -- sysfs parsing ----------------------------------------------------------


def test_parse_policies_from_entries():
    # Arrange
    entries = [
        {"related_cpus": "0 1 2 3", "cpuinfo_max_freq": "1804800"},
        {"related_cpus": "4 5 6", "cpuinfo_max_freq": "2419200"},
        {"related_cpus": "7", "cpuinfo_max_freq": "2841600"},
    ]

    # Act
    policies = parse_policies(entries)

    # Assert
    assert [p.cpus for p in policies] == [[0, 1, 2, 3], [4, 5, 6], [7]]
    assert [p.max_freq_khz for p in policies] == [1804800, 2419200, 2841600]


def test_parse_policies_tolerates_missing_freq():
    """A policy without cpuinfo_max_freq is kept (never dropped silently)."""
    # Arrange
    entries = [{"related_cpus": "0 1", "cpuinfo_max_freq": None}]

    # Act
    policies = parse_policies(entries)

    # Assert
    assert policies[0].cpus == [0, 1]
    assert policies[0].max_freq_khz == 0


def test_parse_policies_skips_unparseable_related_cpus():
    # Arrange
    entries = [{"related_cpus": "", "cpuinfo_max_freq": "1000"},
               {"related_cpus": "0 1", "cpuinfo_max_freq": "2000"}]

    # Act
    policies = parse_policies(entries)

    # Assert
    assert [p.cpus for p in policies] == [[0, 1]]


# -- hexadecimal task mask --------------------------------------------------


def test_cpu_mask_big_cluster():
    # Act / Assert
    assert cpu_mask([4, 5, 6, 7]) == "f0"


def test_cpu_mask_single_cores():
    # Act / Assert
    assert cpu_mask([0]) == "1"
    assert cpu_mask([7]) == "80"


def test_cpu_mask_little_cluster():
    # Act / Assert
    assert cpu_mask([0, 1, 2, 3]) == "f"


def test_cpu_mask_requires_at_least_one_cpu():
    # Act / Assert
    with pytest.raises(CpuSelectionError, match="at least one"):
        cpu_mask([])


def test_cpu_mask_spans_full_64_bit_word():
    # Act / Assert
    assert cpu_mask([63]) == "8000000000000000"
    assert cpu_mask(list(range(64))) == "ffffffffffffffff"


def test_cpu_mask_rejects_beyond_64_bits():
    """Above one machine word the mask notation diverges between the toybox
    and util-linux taskset builds -> refuse rather than guess."""
    # Act / Assert
    with pytest.raises(CpuSelectionError, match="64"):
        cpu_mask([64])
