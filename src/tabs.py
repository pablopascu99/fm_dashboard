import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils import load_models, preprocess_data, predict

metric_subgroups = {
    "Network": [
        ["RxBytes", "TxBytes"],
        ["TxDiscards", "RxDiscards"], 
        ["TxNonUnicastPkts", "RxNonUnicastPkts"],
        ["TxUnicastPkts", "RxUnicastPkts"]
    ],
    "TCP": [
        ["TcpTxResets"], 
        ["TcpRxSegments", "TcpTxSegments"],
        ["TcpRetransmissions"],
        ["TcpPassiveOpens", "TcpActiveOpens"],
        ["TcpEstabResets", "TcpCurrentConnections"]
    ],
    "UDP": [
        ["UdpRxDatagrams", "UdpTxDatagrams"],
        ["UdpRxErrors", "UdpNoPortsErrors"]
    ],
    "IP": [
        ["IpRxPackets","IpForwardedDatagrams"], 
        ["IpDeliveredPackets", "IpTxRequests"],
        ["IpTxDiscards", "IpRxDiscards"],
        ["IpNoRoutePackets", "IpAddressErrors"]
    ],
    "ICMP": [
        ["IcmpRxMessages", "IcmpRxDestUnreach"],
        ["IcmpTxMessages", "IcmpTxDestUnreach"],
        ["IcmpRxEchoRequests", "IcmpTxEchoReplies"]
    ]
}

# Mapping of original columns to standardized metrics and their units
column_mapping = {
    "ifInOctets11": ("RxBytes", "bps"),
    "ifOutOctets11": ("TxBytes", "bps"),
    "ifoutDiscards11": ("TxDiscards", "pkts"),
    "ifInUcastPkts11": ("RxUnicastPkts", "pkts"),
    "ifInNUcastPkts11": ("RxNonUnicastPkts", "pkts"),
    "ifInDiscards11": ("RxDiscards", "pkts"),
    "ifOutUcastPkts11": ("TxUnicastPkts", "pkts"),
    "ifOutNUcastPkts11": ("TxNonUnicastPkts", "pkts"),
    "tcpOutRsts": ("TcpTxResets", "pkts"),
    "tcpInSegs": ("TcpRxSegments", "pkts"),
    "tcpOutSegs": ("TcpTxSegments", "pkts"),
    "tcpPassiveOpens": ("TcpPassiveOpens", "conn"),
    "tcpRetransSegs": ("TcpRetransmissions", "pkts"),
    "tcpCurrEstab": ("TcpCurrentConnections", "conn"),
    "tcpEstabResets": ("TcpEstabResets", "conn"),
    "tcpActiveOpens": ("TcpActiveOpens", "conn"),
    "udpInDatagrams": ("UdpRxDatagrams", "pkts"),
    "udpOutDatagrams": ("UdpTxDatagrams", "pkts"),
    "udpInErrors": ("UdpRxErrors", "pkts"),
    "udpNoPorts": ("UdpNoPortsErrors", "pkts"),
    "ipInReceives": ("IpRxPackets", "pkts"),
    "ipInDelivers": ("IpDeliveredPackets", "pkts"),
    "ipOutRequests": ("IpTxRequests", "pkts"),
    "ipOutDiscards": ("IpTxDiscards", "pkts"),
    "ipInDiscards": ("IpRxDiscards", "pkts"),
    "ipForwDatagrams": ("IpForwardedDatagrams", "pkts"),
    "ipOutNoRoutes": ("IpNoRoutePackets", "pkts"),
    "ipInAddrErrors": ("IpAddressErrors", "pkts"),
    "icmpInMsgs": ("IcmpRxMessages", "pkts"),
    "icmpInDestUnreachs": ("IcmpRxDestUnreach", "pkts"),
    "icmpOutMsgs": ("IcmpTxMessages", "pkts"),
    "icmpOutDestUnreachs": ("IcmpTxDestUnreach", "pkts"),
    "icmpInEchos": ("IcmpRxEchoRequests", "pkts"),
    "icmpOutEchoReps": ("IcmpTxEchoReplies", "pkts")
}

def render_metrics_dashboard(df):
    """Renders the network metrics dashboard tab."""
    st.title("Network Metrics Dashboard")

    # Create columns for filters
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    # Metric type selector
    with col1:
        st.subheader("Metric Types")
        metric_types = ["Network", "TCP", "UDP", "IP", "ICMP"]
        selected_type = st.selectbox("Select a metric type", metric_types, key="metric_selectbox", help="Select the type of metric you want to visualize", label_visibility="collapsed")

    # Time interval selector
    with col3:
        st.subheader("Filter Time")
        start_time, end_time = st.select_slider(
            "Select a time interval",
            options=pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M').tolist(),
            value=(df['timestamp'].min().strftime('%Y-%m-%d %H:%M'), df['timestamp'].max().strftime('%Y-%m-%d %H:%M')),
            format_func=lambda x: x,
            key="time_slider",
            help="Select the time interval to filter the data",
            label_visibility="collapsed"
        )

    # Filter data based on selected time interval
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    df_filtered = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]

    # Display metrics for the selected type
    if selected_type:
        st.header(f"{selected_type} Metrics")
        subgroups = metric_subgroups.get(selected_type, [])
        col1, col2 = st.columns(2)
        col = col1

        for subgroup in subgroups:
            fig = go.Figure()
            for metric in subgroup:
                if metric in df_filtered.columns:
                    # Get the metric unit
                    unit = next((column_mapping[key][1] for key in column_mapping if column_mapping[key][0] == metric), "")
                    # Add trace with unit in legend
                    fig.add_trace(go.Scatter(
                        x=df_filtered['timestamp'], 
                        y=df_filtered[metric],
                        mode='lines',
                        name=f"{metric} ({unit})",  # Add unit to legend
                        showlegend=True
                    ))
            # Add title to the chart with included metrics
            metrics_in_group = ", ".join([metric for metric in subgroup if metric in df_filtered.columns])
            fig.update_layout(
                title=f"{metrics_in_group}",
                legend_title="Metrics"
            )
            col.plotly_chart(fig)
            col = col2 if col == col1 else col1

def render_anomaly_detection():
    """Renders the anomaly detection tab."""
    st.title("Network Anomaly Detection")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "anomaly" in df.columns:
            true_labels = df["anomaly"]
            df = df.drop(columns=["anomaly"])
        else:
            true_labels = None

        models, encoder, scaler = load_models()
        X = preprocess_data(df, scaler)
        predictions = predict(models, X, encoder)

        st.write("### Model Predictions:")
        results = pd.DataFrame(predictions)
        
        # Add the original column if true_labels is available
        if true_labels is not None:
            results["originalAnomaly"] = true_labels.values

        st.dataframe(results)