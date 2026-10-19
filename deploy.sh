#!/bin/bash
# ============================================
# DEPLOY TO AZURE CONTAINER APPS
# ============================================
# Deploys the image GitHub Actions built for the current commit
# (.github/workflows/image.yml) to an Azure Container App that scales to
# zero when idle. Paused reviews live in a Postgres database you provide,
# such as a free Neon project. The first run creates the Azure resources;
# later runs roll the app to the new image.
#
# Within Azure's monthly free allowance this costs nothing: no registry
# (GitHub hosts the image) and no database (yours). The first request
# after a quiet spell takes about 25 seconds while the app starts.
#
# The API requires an X-API-Key header. The key is generated on the first
# run and kept in $SECRETS_FILE (mode 600). Nothing secret is printed.
#
# Prerequisites:
#   1. Azure CLI installed (brew install azure-cli) and logged in (az login)
#   2. OPENAI_API_KEY exported
#   3. DATABASE_URL in ~/.config/doc-agent/neon.env (mode 600): a direct,
#      not pooled, connection string
#   4. The commit pushed to main and its image build finished
#
# Usage: ./deploy.sh          (LOCATION=westus2 ./deploy.sh for another region)

set -euo pipefail

: "${OPENAI_API_KEY:?Export OPENAI_API_KEY first}"

LOCATION="${LOCATION:-eastus}"
RESOURCE_GROUP="doc-agent-rg"
APP_ENV="doc-agent-env"
APP_NAME="doc-agent"
IMAGE_REPO="ghcr.io/victorblash/readback"
SECRETS_FILE="${HOME}/.config/doc-agent/azure.env"
DATABASE_FILE="${HOME}/.config/doc-agent/neon.env"

# --- The database ---
if [ ! -f "$DATABASE_FILE" ]; then
    echo "Missing $DATABASE_FILE with DATABASE_URL='postgresql://...'" >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$DATABASE_FILE"
: "${DATABASE_URL:?DATABASE_URL is empty in $DATABASE_FILE}"
if [[ "$DATABASE_URL" == *-pooler* ]]; then
    # PostgresSaver relies on session features a transaction pooler drops.
    echo "DATABASE_URL is a pooled connection string; use the direct one." >&2
    exit 1
fi

# --- The image for this commit ---
SHA=$(git rev-parse HEAD)
git fetch --quiet origin main
if ! git merge-base --is-ancestor "$SHA" origin/main; then
    echo "Commit ${SHA:0:7} isn't on origin/main yet. Push it first." >&2
    exit 1
fi
# The image is built only when its inputs change, so deploy the newest
# commit whose build succeeded at or before this one.
BUILDS=$(gh run list --workflow image.yml --branch main --status success --limit 50 \
    --json headSha --jq '.[].headSha')
BUILT_SHA=""
for candidate in $BUILDS; do
    if git merge-base --is-ancestor "$candidate" "$SHA" 2>/dev/null; then
        BUILT_SHA="$candidate"
        break
    fi
done
if [ -z "$BUILT_SHA" ]; then
    echo "No successful image build found for ${SHA:0:7} or earlier. Check the Actions tab." >&2
    exit 1
fi
IMAGE="$IMAGE_REPO:$BUILT_SHA"
echo "Deploying image built from ${BUILT_SHA:0:7} to $LOCATION"

# --- The API key, generated once ---
if [ ! -f "$SECRETS_FILE" ]; then
    mkdir -p "$(dirname "$SECRETS_FILE")"
    chmod 700 "$(dirname "$SECRETS_FILE")"
    (umask 077; echo "DOC_AGENT_API_KEY=$(openssl rand -hex 32)" > "$SECRETS_FILE")
fi
# shellcheck source=/dev/null
source "$SECRETS_FILE"

echo "Preparing the subscription..."
az extension add --name containerapp --upgrade --only-show-errors
for provider in Microsoft.App Microsoft.OperationalInsights; do
    az provider register --namespace "$provider" --wait --only-show-errors
done

echo "Resource group..."
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" -o none

echo "Container Apps environment (first run takes a few minutes)..."
if ! az containerapp env show --resource-group "$RESOURCE_GROUP" --name "$APP_ENV" -o none 2>/dev/null; then
    az containerapp env create --resource-group "$RESOURCE_GROUP" --name "$APP_ENV" --location "$LOCATION" -o none
fi

# Secrets are Container Apps secrets referenced with secretref:, never
# literal env vars, which anyone who can read the app's configuration
# could see. For production, keep them in Key Vault (keyvaultref:).
# Limitation: the values pass through this script's argv while `az` runs,
# so `ps` on this machine can see them for that moment.
SECRETS=(
    "openai-api-key=$OPENAI_API_KEY"
    "database-url=$DATABASE_URL"
    "api-key=$DOC_AGENT_API_KEY"
)
echo "App..."
if ! az containerapp show --resource-group "$RESOURCE_GROUP" --name "$APP_NAME" -o none 2>/dev/null; then
    az containerapp create \
        --resource-group "$RESOURCE_GROUP" \
        --name "$APP_NAME" \
        --environment "$APP_ENV" \
        --image "$IMAGE" \
        --target-port 8000 \
        --ingress external \
        --min-replicas 0 \
        --max-replicas 2 \
        --cpu 1.0 \
        --memory 2.0Gi \
        --secrets "${SECRETS[@]}" \
        --env-vars \
            OPENAI_API_KEY=secretref:openai-api-key \
            DATABASE_URL=secretref:database-url \
            DOC_AGENT_API_KEY=secretref:api-key \
        -o none
else
    az containerapp secret set --resource-group "$RESOURCE_GROUP" --name "$APP_NAME" --secrets "${SECRETS[@]}" -o none
    az containerapp update --resource-group "$RESOURCE_GROUP" --name "$APP_NAME" --image "$IMAGE" -o none
fi

APP_URL=$(az containerapp show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$APP_NAME" \
    --query "properties.configuration.ingress.fqdn" \
    --output tsv)

echo ""
echo "Deployed ${BUILT_SHA:0:7} to https://$APP_URL"
echo "API key: DOC_AGENT_API_KEY in $SECRETS_FILE"
echo ""
echo "Try it (the first request may take ~25 s while the app starts):"
echo "  curl https://$APP_URL/health"
echo "  source $SECRETS_FILE"
echo "  curl -X POST https://$APP_URL/process -H \"X-API-Key: \$DOC_AGENT_API_KEY\" \\"
echo "    -H 'Content-Type: application/json' -d '{\"document_text\": \"INVOICE #123 ...\"}'"
