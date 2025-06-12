import yaml

# Load the YAML file
with open("hw4fortunes.yml", "r", encoding="utf-8") as f:
    fortunes = yaml.safe_load(f)

with open("new.txt", "w", encoding="utf-8") as f:
    # Generate LDIF
    for entry in fortunes:
        id = entry["ID"]
        author = entry["Author"]
        description = entry["Description"]

        f.write(f"dn: cn=fortune-{id},ou=Fortune,dc=51,dc=nasa\n")
        f.write("objectClass: top\n")
        f.write("objectClass: fortune\n")
        f.write(f"cn: fortune-{id}\n")
        f.write(f"id: {id}\n")
        f.write(f"author: {author}\n")
        for line in description.strip().splitlines():
            f.write(f"description: {line.strip()}\n")
        f.write("\n")  # blank line between entries
