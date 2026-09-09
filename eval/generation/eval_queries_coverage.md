# Evaluation Queries Coverage Report

**Total queries**: 30 (30 dataset + 0 manual)

## ORIGIN × OBSERVATION_TYPE (dataset queries)

```
                     part_replacement  operational_anomaly  TOTAL
table                               6                    6     12
prose                               5                    6     11
TOTAL                              13                   14     27
```

## TRACE VARIANT COVERAGE

```
#1  DRC early-exit                        2  (min 2) OK
#2  Follow-up                             3  (min 3) OK
#3  Troubleshoot short                    2  (min 2) OK
#4  Full troubleshoot                    10  (min 8) OK
#5  Full replace_part                    11  (min 8) OK
#6  UnsafeMethodGate→troubleshoot         1  (min 1) OK
#7  UnsafeMethodGate→replace_part         1  (min 1) OK
                                        Total: 30
```

## SECTION_SYSTEM DISTRIBUTION (dataset queries)

```
  Landing Gear: 3
  Leveling & Weighing: 2
  Flight Controls: 2
  Towing & Taxiing: 1
  Landing Gear (6.3.7): 1
  Fuel (Ch 28): 1
  Flight Controls (6.3.6): 1
  Exhaust (Ch 78): 1
  Engine (6.3.10): 1
  Electrical Power Systems (6.3.17): 1
  Engine Controls (Ch 76): 1
  Indicating / Recording Systems (Ch 31): 1
  Communications (Ch 23): 1
  Oil (Ch 79): 1
  Navigation / Attitude and Direction (Ch 34): 1
  Navigation / Flight Environmental Data (Ch 34): 1
  Empennage: 1
  Servicing: 1
  Equipment & Furnishings: 1
  Wheel & Brake: 1
  Electrical: 1
  Parking & Mooring: 1
  Time Limits & Maintenance Checks: 1
  Electrical Power Systems: 1
  Fuel System: 1
  Exhaust System: 1

Unique systems: 26 (target: as many as possible, max 4 per system)
```

## SAFETY COVERAGE

```
Queries with safety_expected=true:  13 (min 6)
Queries with safety_expected=false: 17
Status: OK
```

## MANUALLY CRAFTED QUERIES NEEDED

```
DRC early-exit queries:       0
Follow-up queries:            0
UnsafeMethodGate queries:     0
Total manual queries needed:  0
```

## QUERY LIST

| query_id | origin | obs_type | section_system | variant | safety | source_index |
|----------|--------|----------|----------------|---------|--------|--------------|
| EQ-001 | prose | operational_anomal | Leveling & Weighing | #3 | N | EVAL-PROSE-069 |
| EQ-002 | prose | operational_anomal | Towing & Taxiing | #3 | N | EVAL-PROSE-131 |
| EQ-003 | table | part_replacement | Landing Gear (6.3.7) | #5 | Y | EVAL-148 |
| EQ-004 | table | part_replacement | Fuel (Ch 28) | #5 | Y | EVAL-137 |
| EQ-005 | table | part_replacement | Flight Controls (6.3.6) | #5 | N | EVAL-140 |
| EQ-006 | table | part_replacement | Exhaust (Ch 78) | #5 | Y | EVAL-146 |
| EQ-007 | table | part_replacement | Engine (6.3.10) | #5 | Y | EVAL-122 |
| EQ-008 | table | part_replacement | Electrical Power Systems  | #5 | Y | EVAL-115 |
| EQ-009 | table | operational_anomal | Engine Controls (Ch 76) | #4 | N | EVAL-029 |
| EQ-010 | table | operational_anomal | Indicating / Recording Sy | #4 | N | EVAL-114 |
| EQ-011 | table | operational_anomal | Communications (Ch 23) | #4 | N | EVAL-035 |
| EQ-012 | table | operational_anomal | Oil (Ch 79) | #4 | Y | EVAL-026 |
| EQ-013 | table | operational_anomal | Navigation / Attitude and | #4 | N | EVAL-058 |
| EQ-014 | table | operational_anomal | Navigation / Flight Envir | #4 | N | EVAL-088 |
| EQ-015 | prose | part_replacement | Empennage | #5 | N | EVAL-PROSE-079 |
| EQ-016 | prose | part_replacement | Servicing | #5 | N | EVAL-PROSE-039 |
| EQ-017 | prose | part_replacement | Flight Controls | #5 | N | EVAL-PROSE-120 |
| EQ-018 | prose | part_replacement | Equipment & Furnishings | #5 | N | EVAL-PROSE-046 |
| EQ-019 | prose | part_replacement | Wheel & Brake | #5 | Y | EVAL-PROSE-004 |
| EQ-020 | prose | operational_anomal | Electrical | #4 | Y | EVAL-PROSE-001 |
| EQ-021 | prose | operational_anomal | Landing Gear | #4 | Y | EVAL-PROSE-002 |
| EQ-022 | prose | operational_anomal | Parking & Mooring | #4 | N | EVAL-PROSE-137 |
| EQ-023 | prose | operational_anomal | Time Limits & Maintenance | #4 | N | EVAL-PROSE-106 |
| EQ-024 | manual | operational_anomal | Electrical Power Systems | #1 | Y | MANUAL-DRC-1 |
| EQ-025 | manual | part_replacement | Landing Gear | #1 | Y | MANUAL-DRC-2 |
| EQ-026 | manual | n/a | Leveling & Weighing | #2 | N | MANUAL-FOLLOWUP-1 |
| EQ-027 | manual | n/a | Flight Controls | #2 | N | MANUAL-FOLLOWUP-2 |
| EQ-028 | manual | n/a | Fuel System | #2 | N | MANUAL-FOLLOWUP-3 |
| EQ-029 | manual | operational_anomal | Landing Gear | #6 | Y | MANUAL-UNSAFE-1 |
| EQ-030 | manual | part_replacement | Exhaust System | #7 | Y | MANUAL-UNSAFE-2 |