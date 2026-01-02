import tomllib
import asyncio
import aiohttp_oauthlib as aioauth
import os
from typing import Any
import math

auth_base_url = "https://oauth.onshape.com/oauth/authorize"
token_url = "https://oauth.onshape.com/oauth/token"
redirect_url = "http://localhost:5000/token"

grav_accel = 9.81  # m/s^2
air_density = 1.204  # kg/m^3

os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"


def load_auth() -> dict[str, dict[str, str]]:
    with open("auth.toml", "rb") as f:
        auth = tomllib.load(f)

        match auth:
            case {"client": {"id": str(), "secret": str()}}:
                return auth
            case _:
                raise ValueError("Invalid auth structure")


def load_config() -> dict[str, dict[str, str | bool | int]]:
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)

        match config:
            case {
                "request": {
                    "assembly_url": str(),
                },
                "assembly": {
                    "include_mate_features": bool(),
                    "include_non_solids": bool(),
                    "include_mate_connectors": bool(),
                    "exclude_suppressed": bool(),
                },
                "parts": {
                    "rollback_bar_index": int(),
                    "infer_metadata_owner": bool(),
                    "use_mass_property_overrides": bool(),
                },
            }:
                return config
            case _:
                raise ValueError("Invalid configuration.")


def parse_url(url: str | bool | int) -> list[str]:
    url = str(url).replace("documents", "d")

    d_sep = "/d/"
    wvm_sep = "/m/"
    e_sep = "/e/"

    if url.index("/w/") > -1:
        wvm_sep = "/w/"

    d_pos = url.index(d_sep)
    wvm_pos = url.index(wvm_sep)
    e_pos = url.index(e_sep)

    base = url[:d_pos]
    d = url[d_pos:wvm_pos]
    wvm = url[wvm_pos:e_pos]
    e = url[e_pos:]

    return [base + "/api/v12/", d, wvm, e]


def snake_to_camel(string: str) -> str:
    segs = string.split("_")
    return segs[0] + "".join(seg.capitalize() for seg in segs[1:])


def snake_keys_to_camel(dictionary: dict[str, str | bool | int]):
    return {
        snake_to_camel(key): str(dictionary[key]).lower() for key in dictionary.keys()
    }


async def fetch(
    session: aioauth.OAuth2Session, method: str, url: str, params: dict[str, str]
) -> dict[str, Any]:
    async with session.request(method, url, params=params) as resp:
        return await resp.json()


async def explore_assembly(
    session: aioauth.OAuth2Session,
    base: str,
    d: str,
    wvm: str,
    e: str,
    assembly_params: dict[str, str],
    parts_params: dict[str, str],
    tasks: asyncio.TaskGroup,
):
    data = await fetch(
        session,
        "GET",
        f"{base}assemblies{d}{wvm}{e}",
        assembly_params,
    )

    all_instances: list[Any] = []
    parts: dict[str, Any] = {}

    all_instances = data["rootAssembly"]["instances"]
    for assembly in data["subAssemblies"]:
        all_instances.extend(assembly["instances"])

    for component in all_instances:
        if component["type"] == "Part":
            parts.update(
                {
                    component["id"]: tasks.create_task(
                        fetch(
                            session,
                            "GET",
                            f"{base}parts/d/{component['documentId']}/m/{
                                component['documentMicroversion']
                            }/e/{component['elementId']}/partid/{
                                component['partId']
                            }/massproperties",
                            parts_params,
                        )
                    )
                }
            )

    return data, parts


def transform_vector(matrix: list[float], vector: list[float]) -> list[float]:
    # scale
    vector = [matrix[15] * vector[i] for i in range(3)]
    # rotate
    vector = [sum([matrix[i + j] * vector[j] for j in range(3)]) for i in [0, 4, 8]]
    # transform
    vector = [matrix[3 + 4 * i] + vector[i] for i in range(3)]
    return vector


def evaluate_parts(
    top_level: dict[str, Any], parts: dict[str, Any]
) -> tuple[float, float, list[float], list[float], float, list[float]]:
    total_mass: float = 0
    total_volume: float = 0
    centre_grav: list[float] = [0, 0, 0]
    centre_buoy: list[float] = [0, 0, 0]
    instability: float = 0
    axis: list[float] = [0, 0, 0]

    for part in parts.keys():
        for occurrence in top_level["rootAssembly"]["occurrences"]:
            if occurrence["path"][-1] == part:
                body = next(iter(parts[part]["bodies"].values()))
                transform = occurrence["transform"]

                mass: float = body["mass"][0]
                volume: float = body["volume"][0]
                centroid: list[float] = body["centroid"][0:3]

                total_mass += mass
                total_volume += volume

                position = transform_vector(transform, centroid)

                for i in range(3):
                    centre_grav[i] += mass * position[i]
                    centre_buoy[i] += volume * position[i]

    for i in range(3):
        centre_grav[i] /= total_mass
        centre_buoy[i] /= total_volume

    vec_gb = [centre_buoy[i] - centre_grav[i] for i in range(3)]
    mag_gb: float = (sum([(vec_gb[i]) ** 2 for i in range(3)])) ** 0.5
    instability = math.acos(vec_gb[2] / mag_gb) * 180 / math.pi

    axis = [vec_gb[1], -vec_gb[0], 0]

    return total_mass, total_volume, centre_grav, centre_buoy, instability, axis


def format_list(fmt: str, items: list[float]) -> str:
    splt = [fmt for _ in items]
    fmt = "[" + ", ".join(splt) + "]"
    return fmt.format(*items)


async def main():
    config = load_config()
    parsed_url = parse_url(config["request"]["assembly_url"])

    assembly_params = snake_keys_to_camel(config["assembly"])
    parts_params = snake_keys_to_camel(config["parts"])

    auth = load_auth()

    oauth = aioauth.OAuth2Session(
        client_id=auth["client"]["id"],
        redirect_uri=redirect_url,
    )

    auth_url, state = oauth.authorization_url(auth_base_url)
    print("Please authenticate at: ", auth_url)
    auth_resp = input("Enter callback url (localhost): ")

    token = await oauth.fetch_token(
        token_url=token_url,
        client_secret=auth["client"]["secret"],
        authorization_response=auth_resp,
    )

    top_level: dict[str, Any] = {}
    parts: dict[str, Any] = {}

    async with asyncio.TaskGroup() as tasks:
        top_level, parts = await explore_assembly(
            oauth,
            parsed_url[0],
            parsed_url[1],
            parsed_url[2],
            parsed_url[3],
            assembly_params,
            parts_params,
            tasks,
        )
        print(f"\nFound {len(parts)} parts. Evaluating...")

    for part in parts.keys():
        parts[part] = parts[part].result()

    total_mass, total_volume, centre_grav, centre_buoy, instability, axis = (
        evaluate_parts(top_level, parts)
    )

    print(
        "\nmass: {:.3f} kg\nvolume: {:.3e} m^3\ncentre_grav: {} m\ncentre_buoy: {} m\ninstability: {:.2f} deg\naxis: {} m".format(
            total_mass,
            total_volume,
            format_list("{:.3e}", centre_grav),
            format_list("{:.3e}", centre_buoy),
            instability,
            format_list("{:.3e}", axis),
        )
    )

    await oauth.close()


if __name__ == "__main__":
    asyncio.run(main())
