---
icon: material/lan
---

# Network Configuration

Agora needs inbound TCP on port **49200** per GPU (49200 for GPU 0, 49201 for GPU 1, …). Where and how you open that port depends on where the node lives. This guide covers every provider in [Cloud Options](cloud.md) plus Personal computer and WSL2.

## At a glance

| Environment | Where to configure | External port |
|---|---|---|
| **AWS** | Security Group → inbound rule | Same as internal (49200) |
| **GCP** | VPC network → firewall rule | Same as internal (49200) |
| **RunPod** | Pod settings → Expose TCP Ports | **Random** — read from Connect panel |
| **Vast.ai** | Docker `-p` flag in image config | **Random** — read from IP Port Info |
| **Tensordock** *(Distributed)* | Port Forwarding during provisioning | **Random** — read from instance details |
| **Lambda Labs** | Cloud dashboard → Firewall | Same as internal (49200) |
| **Personal / home** | Local firewall **and** router port forward | Same as internal (49200) |
| **WSL2** | `netsh portproxy` + Windows firewall | Same as internal (49200) |

!!! warning "Random external ports need `--announce_port`"
    RunPod, Vast.ai, and Tensordock assign a random external port that differs from the internal `49200`. Pass both `--host_port 49200` and `--announce_port <external>` when launching Agora so peers reach you at the right address.

---

## AWS

1. Navigate to your EC2 instance and click on the **Security Group** attached to it (under the **Security** tab).
2. Click **Inbound rules** → **Edit inbound rules**.
3. Click **Add rule** and configure:
    - **Type:** Custom TCP
    - **Port range:** `49200`
    - **Source:** `0.0.0.0/0` (allow traffic from any source)

<img class="docs-screenshot" src="../../../images/aws-inbound-rules.png" alt="AWS Inbound Rules" width="800">

---

## GCP

1. Go to **VPC network** → **Firewall** → **Create firewall rule**.
2. Configure the rule:
    - **Target:** "All instances in the network" — or use "Specified target tags" and tag your instance.
    - **Source IPv4 ranges:** `0.0.0.0/0`
    - **Protocols and ports:** Select "Specified protocols and ports"
    - **TCP:** `49200`

<img class="docs-screenshot" src="../../../images/gcp-inbound-rules.png" alt="GCP Inbound Rules" width="800">

---

## RunPod

RunPod assigns a **random external port mapping** — you need to expose the internal port and then read back the external one.

1. After deploying a Pod, click the three horizontal lines (Pod Settings) → **Edit Pod**.

    <img class="docs-screenshot" src="../../../images/runpod-edit-pod.png" alt="RunPod Edit Pod menu" width="800">

2. Under **Expose TCP Ports** add `49200` and save (this restarts the Pod).

    <img class="docs-screenshot" src="../../../images/runpod-inbound-rules.png" alt="RunPod TCP port mapping" width="600">

3. Once restarted, click **Connect** to see the external port mapping for internal port 49200. In the example below the external port is `17969`.

    <img class="docs-screenshot" src="../../../images/runpod-external-port.png" alt="RunPod external port mapping" width="600">

4. Pass both ports when launching Agora:

    ```bash
    --host_port 49200      # internal port the library listens on
    --announce_port 17969  # external port other peers connect to
    ```

---

## Vast.ai

Vast.ai exposes ports via Docker's `-p` flag in the instance template. The internal port you request gets mapped to a **random external port** on the host's public IP.

1. **In the instance template** — before launching, open **Edit Image & Config** → in the **Docker create/run options** box, add:

    ```text
    -p 49200:49200
    ```

    Save the template. If you're launching from a search result, click the wrench/pencil icon on the offer card to open this editor before pressing **Rent**.

2. **Launch the instance** in **SSH + TCP** mode (not Jupyter). This ensures the TCP forward is wired up.

3. **Find the assigned external port** — on the **Instances** page, click the **IP & Port Info** button (or the IP-address chip at the top of the instance card). You'll see lines like:

    ```text
    65.130.162.74:33526 -> 49200/tcp
    ```

    Here `33526` is your external port.

    !!! info "Or read it from inside the instance"
        Vast.ai also exports the mapping as the environment variable `VAST_TCP_PORT_49200`. Inside the container: `echo $VAST_TCP_PORT_49200`.

4. **Pass both ports when launching Agora:**

    ```bash
    --host_port 49200      # internal port the library listens on
    --announce_port 33526  # external port from Vast.ai's IP & Port Info
    ```

!!! warning "Pick `-p`, not `EXPOSE`"
    Vast.ai's image-config editor sometimes also surfaces an "Open Ports" / `EXPOSE` list — that field only documents the port in the image metadata, it does **not** publish the port. The `-p` argument in the docker run options is what actually creates the external mapping.

---

## Tensordock

Tensordock's **Distributed Compute** lets you request a specific internal port and then assigns a random external port mapping to it.

1. During provisioning, under the **Port Forwarding** section click **Request Port** and enter `49200`:

    <img class="docs-screenshot" src="../../../images/tensordock-external-port.png" alt="Tensordock Port Forwarding setup" width="800">

2. Once deployed, note the randomly assigned external port:

    <img class="docs-screenshot" src="../../../images/tensordock-forwarded-ports.png" alt="Tensordock Forwarded Ports" width="600">

3. Pass both ports when launching Agora — `--announce_port` is your assigned external port:

    ```bash
    --host_port 49200      # internal port the library listens on
    --announce_port 10009  # external port other peers connect to
    ```

---

## Lambda Labs

1. Navigate to the **Firewall** page in your Lambda Cloud dashboard.
2. Click **Edit** in the **Inbound Rules** section.
3. Configure the new rule:
    - **Rule type:** Custom TCP
    - **Port range:** `49200`
    - **Source:** `0.0.0.0/0` (allow traffic from any source)

<img class="docs-screenshot" src="../../../images/lambda-inbound-rules.png" alt="Lambda Inbound Rules" width="600">

---

## Personal computer

If the node lives on a laptop or desktop on your home network, two things need to allow `49200/tcp`:

1. **Local firewall** — open inbound `49200/tcp` (Linux: `ufw allow 49200/tcp`; macOS: System Settings → Network → Firewall; Windows: Defender Firewall inbound rule).
2. **Router** — set up port forwarding for `49200/tcp` from the WAN to your machine's LAN IP. The exact location varies by router brand; look for *Port Forwarding*, *Virtual Server*, or *NAT*.

---

## WSL2

Windows + WSL2 adds an extra hop: traffic arrives at Windows and needs a port-proxy into the WSL2 VM.

1. **Enable localhost forwarding** — create `C:\Users\<YourUsername>\.wslconfig` with:

    ```text
    [wsl2]
    localhostforwarding=true
    ```

2. **Configure the port proxy** (PowerShell **as Administrator**). First find the WSL container's IP:

    ```powershell
    ((wsl hostname -I) -split " ")[0]
    ```

    Add the proxy, replacing `<wsl_ip>` with the IP from above:

    ```powershell
    netsh interface portproxy add v4tov4 listenport=49200 listenaddress=0.0.0.0 connectport=49200 connectaddress=<wsl_ip>
    ```

    Open the Windows firewall for the port:

    ```powershell
    netsh advfirewall firewall add rule name="agora" dir=in action=allow protocol=TCP localport=49200
    ```

3. **Router** — set up port forwarding from the WAN to your Windows machine on `49200/tcp`.
4. **Restart** to apply changes.
