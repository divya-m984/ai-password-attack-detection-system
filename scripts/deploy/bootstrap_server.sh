#!/usr/bin/env bash
#
# SERVER ONLY. Do not run this on a development machine.
#
# Prepares a fresh Ubuntu 24.04-class VPS to run this project's containerized
# demonstration: installs Docker from Docker's own apt repository, enables it,
# and creates an unprivileged account to own the checkout and drive Compose.
#
# It refuses to run unless you pass --confirm, and it checks that the machine
# looks like a fresh Ubuntu server first, because "modify every package on this
# host" is not something a script should be able to do to somebody's laptop by
# being invoked with a stray tab-completion.
#
# WHAT IT DOES NOT DO, DELIBERATELY
#
#   * No credential, token, key, or cloud provider API call. Nothing here talks
#     to DigitalOcean or to any other provider.
#   * No change to Git identity, and no Git operation of any kind on the host it
#     runs on beyond installing the `git` package.
#   * No change to sshd_config. It INSPECTS the SSH configuration and warns, and
#     it never edits it: a bootstrap script that rewrites sshd on a machine you
#     are connected to over SSH is a script that can lock you out of it. Password
#     authentication is never enabled, and key authentication is never weakened.
#   * No firewall change unless you ask for one with --with-ufw, and then only
#     ports 22, 80 and 443 -- SSH first, so enabling the firewall cannot end the
#     session that is enabling it. The cloud provider's own firewall is the
#     primary control; see docs/deployment.md.
#   * It does not clone the repository, build an image, or start anything. Those
#     are the deployment user's steps, run deliberately, and documented.
#
# ABOUT THE DOCKER GROUP
#
# The deployment user is added to the `docker` group so it can run Compose
# without sudo. Be clear-eyed about what that is: membership in `docker` grants
# control of a root daemon that can mount the host filesystem into a container.
# It is root-equivalent in practice. It is not a security boundary between the
# deployment user and root -- it is a convenience, and the boundary that matters
# is the one around SSH access to this machine.
#
# Usage:
#
#     sudo bash scripts/deploy/bootstrap_server.sh --confirm
#     sudo bash scripts/deploy/bootstrap_server.sh --confirm --user pad --with-ufw

set -euo pipefail

DEPLOY_USER="pad"
CONFIRMED=false
WITH_UFW=false

die() {
    echo "bootstrap: $*" >&2
    exit 1
}

note() {
    echo "==> $*"
}

warn() {
    echo "!!  $*" >&2
}

usage() {
    cat <<'USAGE'
usage: bootstrap_server.sh --confirm [--user NAME] [--with-ufw]

  --confirm     required. Acknowledges that this modifies the whole host.
  --user NAME   the unprivileged deployment account to create (default: pad).
  --with-ufw    also allow 22, 80 and 443 in ufw and enable it. Off by default;
                the cloud firewall is the primary control.

Run as root on a fresh Ubuntu 24.04-class server, and nowhere else.
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --confirm) CONFIRMED=true ;;
        --with-ufw) WITH_UFW=true ;;
        --user)
            shift
            [ $# -gt 0 ] || die "--user needs a value"
            DEPLOY_USER="$1"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *) die "unknown argument: $1" ;;
    esac
    shift
done

# ---------------------------------------------------------------------------
# Refuse anything that is not the machine this is for.
# ---------------------------------------------------------------------------

if [ "$CONFIRMED" != true ]; then
    usage
    die "refusing to run without --confirm"
fi

# The "is this the right machine" checks come before the "are you root" check on
# purpose: a developer who runs this by accident should be told that their laptop
# is the wrong machine, not that they need sudo.
[ -r /etc/os-release ] || die "no /etc/os-release; this is not the Ubuntu server this expects"
# shellcheck disable=SC1091  # a distribution file, not a project file
. /etc/os-release
[ "${ID:-}" = "ubuntu" ] || die "expected Ubuntu, found ID=${ID:-unknown}"

case "${VERSION_ID:-}" in
    24.*|25.*) ;;
    *) warn "expected Ubuntu 24.04-class, found ${VERSION_ID:-unknown}; continuing" ;;
esac

# A crude but effective "is this somebody's workstation" check. A server does
# not have a display manager, and this script has no business on a machine that
# does.
if [ -d /usr/share/xsessions ] || [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; then
    die "this machine has a graphical session; bootstrap_server.sh is for servers only"
fi

[ "$(id -u)" -eq 0 ] || die "run this as root (sudo bash $0 --confirm)"

[ "$DEPLOY_USER" != "root" ] || die "the deployment user must not be root"
printf '%s' "$DEPLOY_USER" | grep -Eq '^[a-z_][a-z0-9_-]{0,31}$' \
    || die "invalid user name: $DEPLOY_USER"

note "Ubuntu ${VERSION_ID:-?}, deployment user '${DEPLOY_USER}'."

# ---------------------------------------------------------------------------
# Docker, from Docker's own repository.
#
# Not the distribution's docker.io package: this deployment pins its base image
# by digest and uses Compose v2 features, and the upstream repository is what
# keeps the engine and the compose plugin versioned together.
# ---------------------------------------------------------------------------

export DEBIAN_FRONTEND=noninteractive

note "Installing prerequisites."
apt-get update -qq
apt-get install -y -qq ca-certificates curl gnupg git

if [ ! -f /etc/apt/keyrings/docker.asc ]; then
    note "Adding Docker's apt repository."
    install -m 0755 -d /etc/apt/keyrings
    # Over HTTPS to Docker's own host, verified by the system CA store. The key
    # is fetched rather than embedded here: a key pinned in a repository file is
    # a key nobody rotates.
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    chmod a+r /etc/apt/keyrings/docker.asc

    printf 'deb [arch=%s signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu %s stable\n' \
        "$(dpkg --print-architecture)" \
        "${UBUNTU_CODENAME:-${VERSION_CODENAME:-noble}}" \
        > /etc/apt/sources.list.d/docker.list
    apt-get update -qq
else
    note "Docker's apt repository is already configured."
fi

note "Installing Docker Engine, the buildx plugin, and the Compose plugin."
apt-get install -y -qq \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin

note "Enabling the Docker service."
systemctl enable --now docker

docker --version
docker compose version

# ---------------------------------------------------------------------------
# The unprivileged deployment account.
# ---------------------------------------------------------------------------

if id -u "$DEPLOY_USER" >/dev/null 2>&1; then
    note "User '${DEPLOY_USER}' already exists; leaving it alone."
else
    note "Creating '${DEPLOY_USER}'."
    # No password is set, and none may be: this account is reachable by SSH key
    # only. `--disabled-password` leaves the password field locked.
    adduser --disabled-password --gecos "" "$DEPLOY_USER"
fi

note "Adding '${DEPLOY_USER}' to the docker group (root-equivalent -- see the header)."
usermod -aG docker "$DEPLOY_USER"

# Give the new account the same SSH keys root was reached with, so it is usable
# immediately without ever enabling password authentication. Copied, never
# generated: this script creates no key material.
deploy_home="$(getent passwd "$DEPLOY_USER" | cut -d: -f6)"
if [ -f /root/.ssh/authorized_keys ] && [ ! -s "${deploy_home}/.ssh/authorized_keys" ]; then
    note "Copying root's authorized_keys to '${DEPLOY_USER}'."
    install -d -m 0700 -o "$DEPLOY_USER" -g "$DEPLOY_USER" "${deploy_home}/.ssh"
    install -m 0600 -o "$DEPLOY_USER" -g "$DEPLOY_USER" \
        /root/.ssh/authorized_keys "${deploy_home}/.ssh/authorized_keys"
else
    warn "No authorized_keys copied. Install a public key for '${DEPLOY_USER}' before"
    warn "you disable root SSH, or you will lock yourself out."
fi

# ---------------------------------------------------------------------------
# SSH: inspected, never edited.
# ---------------------------------------------------------------------------

note "Checking the SSH configuration (read-only; nothing here edits it)."
if command -v sshd >/dev/null 2>&1 && sshd -T >/dev/null 2>&1; then
    effective="$(sshd -T 2>/dev/null || true)"
    if printf '%s\n' "$effective" | grep -qi '^passwordauthentication yes'; then
        warn "sshd accepts password authentication. Turn it off by hand:"
        warn "    PasswordAuthentication no   in /etc/ssh/sshd_config.d/, then"
        warn "    systemctl reload ssh"
        warn "Confirm key-based login works in a SECOND session before you do."
    else
        note "Password authentication is already off. Good."
    fi
    if printf '%s\n' "$effective" | grep -qi '^permitrootlogin yes'; then
        warn "sshd permits root login. Consider 'PermitRootLogin prohibit-password'"
        warn "once '${DEPLOY_USER}' can log in with a key."
    fi
else
    warn "Could not read the effective sshd configuration; check it by hand."
fi

# ---------------------------------------------------------------------------
# Optional host firewall. Off unless asked for, and never more than three ports.
# ---------------------------------------------------------------------------

if [ "$WITH_UFW" = true ]; then
    command -v ufw >/dev/null 2>&1 || apt-get install -y -qq ufw
    note "Allowing 22, 80 and 443 in ufw."
    # SSH first and unconditionally, so enabling the firewall cannot end the
    # session that enabled it.
    ufw allow 22/tcp
    ufw allow 80/tcp
    ufw allow 443/tcp
    ufw --force enable
    ufw status verbose
else
    note "Leaving the host firewall alone (--with-ufw to configure it)."
    note "Configure the cloud firewall instead: 22 from your address, 80 and 443"
    note "from anywhere, everything else denied. See docs/deployment.md."
fi

# ---------------------------------------------------------------------------
# What happens next, by hand.
# ---------------------------------------------------------------------------

cat <<NEXT

Bootstrap complete.

Log in as '${DEPLOY_USER}' -- a new session, so the docker group takes effect --
and continue by hand:

  git clone https://github.com/<owner>/ai-password-attack-detection-system.git
  cd ai-password-attack-detection-system
  git checkout <reviewed-tag-or-commit>
  git rev-parse HEAD          # record this; it is what you deployed

  cp .env.deploy.example .env.deploy
  \$EDITOR .env.deploy         # hostname, publish specification, routing policy

  docker compose --env-file .env.deploy \\
    -f compose.yaml -f compose.deploy.yaml up -d --build

The first start runs the preparation job before the API comes up. It trains the
champion offline and takes roughly a minute plus build time; nothing is ever
fitted inside the serving process. Full procedure in docs/deployment.md.

NEXT
