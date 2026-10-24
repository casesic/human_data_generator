"""Editor scene defaults and person-add behavior."""

from human_data_generator.components import describe


def default_scene(environment="city"):
    if environment == "city":
        people = [
            dict(
                person_id=1,
                body="M01",
                motion="APD6526",
                clothing="M01_simple",
                hair="M01_groom",
                motion_start=0,
                route_xy_m=[[0, 77.96], [0, 92]],
                speed_m_s=1.2,
                start_progress_m=0,
            )
        ]
        rigs = [
            dict(rig_id="rig_side", origin_m=[6.035128805859624, 75.97759765371869, 0], yaw_deg=298)
        ]
    elif environment == "riverfront":
        people = [
            dict(
                person_id=1,
                body="M01",
                motion="APD6526",
                clothing="M01_simple",
                hair="M01_groom",
                motion_start=0,
                route_xy_m=[[0, 1.8], [0, 14]],
                speed_m_s=1.2,
                start_progress_m=0,
            )
        ]
        rigs = [
            dict(rig_id="rig_edge", origin_m=[0, -0.0676685317844754, 0], yaw_deg=90.70578757417066)
        ]
    else:
        raise ValueError("unknown_environment")
    return dict(
        schema="human_scene_v2",
        scene_id=environment + "_example",
        environment=environment,
        seed=20261023,
        fps=8,
        frame_count=16,
        people=people,
        rigs=rigs,
    )


def additional_person(scene, person_id):
    if type(person_id) is not int or person_id < 1:
        raise ValueError("invalid_person_id")
    inventory = describe()
    available = {
        k: [row["id"] for row in inventory[k] if row["status"] == "ready"]
        for k in ("bodies", "motions", "clothing", "hair")
    }
    if any(not v for v in available.values()):
        raise ValueError("no_available_components")
    base = default_scene(scene["environment"])["people"][0]
    item = dict(base, person_id=person_id)
    for kind, field in (
        ("bodies", "body"),
        ("motions", "motion"),
        ("clothing", "clothing"),
        ("hair", "hair"),
    ):
        item[field] = available[kind][(person_id - 1) % len(available[kind])]
    shift = (person_id - 1) * 1.2
    item["route_xy_m"] = [[x + shift, y] for x, y in base["route_xy_m"]]
    return item
