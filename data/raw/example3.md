# Network Connectivity Troubleshooting Guide

## Common Connectivity Issues

This guide provides steps to diagnose and resolve common network connectivity problems that users might encounter.

## Basic Troubleshooting Steps

### 1. Verify Physical Connections
- Ensure all network cables are properly connected
- Check that network devices (routers, switches) are powered on
- Look for indicator lights on network interfaces (should be green or amber)

### 2. Restart Network Devices
- Restart your computer
- Power cycle your router/modem (unplug for 30 seconds, then plug back in)
- Wait 2-3 minutes for devices to fully initialize

### 3. Check Wi-Fi Settings
- Verify you're connected to the correct Wi-Fi network
- Try forgetting the network and reconnecting
- Check if airplane mode is enabled (should be off)

### 4. Run Network Diagnostics
- **Windows**: Run Network Troubleshooter (Settings > Network & Internet > Status > Network troubleshooter)
- **Mac**: Run Wireless Diagnostics (hold Option key and click Wi-Fi icon)
- **Linux**: Use commands like `ping`, `ifconfig`, and `traceroute`

## Advanced Troubleshooting

### Check IP Configuration
Run the appropriate command for your operating system:
- **Windows**: Open Command Prompt and type `ipconfig /all`
- **Mac/Linux**: Open Terminal and type `ifconfig` or `ip addr`

Verify that:
- You have a valid IP address (not 169.254.x.x which indicates APIPA)
- Subnet mask is correct (typically 255.255.255.0)
- Default gateway is present

### Test Basic Connectivity
1. Ping your default gateway:
   ```
   ping 192.168.1.1
   ```
   (Replace with your actual gateway address)

2. Ping a public DNS server:
   ```
   ping 8.8.8.8
   ```

3. Test DNS resolution:
   ```
   ping google.com
   ```

### Check DNS Settings
- Verify DNS servers are correctly configured
- Try using public DNS servers temporarily (8.8.8.8 and 8.8.4.4 for Google DNS)

## Common Error Messages

### "No Internet Access"
- Check if other devices can connect to the same network
- Verify ISP service status
- Reset your router/modem

### "Limited Connectivity"
- Usually indicates successful connection to local network but no internet access
- Check router's internet connection
- Verify ISP account status

### "DNS Server Not Responding"
- Try alternative DNS servers
- Clear DNS cache:
  - Windows: `ipconfig /flushdns`
  - Mac: `sudo killall -HUP mDNSResponder`
  - Linux: `sudo systemd-resolve --flush-caches`

## When to Escalate
If basic troubleshooting doesn't resolve the issue:
1. Contact your IT department with details of steps already taken
2. Have error messages and diagnostic results ready
3. Be prepared to provide device information (OS, network adapter, etc.)

For critical systems, follow the escalation path defined in your organization's IT support policy.