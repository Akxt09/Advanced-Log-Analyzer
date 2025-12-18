import { useState } from 'react'
import axios from 'axios'
import Plot from 'react-plotly.js'
import { Upload, AlertTriangle, TrendingUp, Activity, Database } from 'lucide-react'

const API_BASE_URL = '/api'

function App() {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)

  const handleFileChange = (e) => {
    setFile(e.target.files[0])
    setError(null)
  }

  const handleAnalyze = async () => {
    if (!file) {
      setError('Please select a log file')
      return
    }

    setLoading(true)
    setError(null)
    setResults(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await axios.post(`${API_BASE_URL}/analyze`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      setResults(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Analysis failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <h1 className="text-3xl font-bold text-gray-900">
            Advanced Log Analyzer
          </h1>
          <p className="text-gray-600 mt-1">
            VAE-based Anomaly Detection + Traffic Forecasting
          </p>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        {/* Upload Section */}
        <div className="card mb-8">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Upload size={24} />
            Upload Log File
          </h2>

          <div className="space-y-4">
            <div className="flex items-center gap-4">
              <input
                type="file"
                accept=".log,.txt,.csv,.tsv"
                onChange={handleFileChange}
                className="block w-full text-sm text-gray-500
                  file:mr-4 file:py-2 file:px-4
                  file:rounded-lg file:border-0
                  file:text-sm file:font-semibold
                  file:bg-blue-50 file:text-blue-700
                  hover:file:bg-blue-100
                  cursor-pointer"
              />
              <button
                onClick={handleAnalyze}
                disabled={loading || !file}
                className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap"
              >
                {loading ? 'Analyzing...' : 'Analyze'}
              </button>
            </div>

            {file && (
              <p className="text-sm text-gray-600">
                Selected: {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
              </p>
            )}

            {error && (
              <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
                {error}
              </div>
            )}

            {loading && (
              <div className="flex items-center gap-3 text-blue-600">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
                <span>Processing log file... This may take a few minutes</span>
              </div>
            )}
          </div>
        </div>

        {/* Results Section */}
        {results && (
          <>
            {/* Stats Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
              <StatCard
                icon={<Database size={24} />}
                title="Total Requests"
                value={results.basic_stats.total_requests.toLocaleString()}
                subtitle={`${results.basic_stats.unique_ips.toLocaleString()} unique IPs`}
              />
              <StatCard
                icon={<AlertTriangle size={24} />}
                title="Error Rate"
                value={`${(results.basic_stats.error_rate * 100).toFixed(2)}%`}
                subtitle="HTTP errors"
              />
              <StatCard
                icon={<Activity size={24} />}
                title="Avg Response"
                value={`${results.basic_stats.avg_response_time.toFixed(0)}ms`}
                subtitle="Time taken"
              />
              <StatCard
                icon={<TrendingUp size={24} />}
                title="Processing Time"
                value={`${results.processing_time_seconds.toFixed(1)}s`}
                subtitle="Analysis completed"
              />
            </div>

            {/* Anomaly Detection Results */}
            {results.anomaly_detection && (
              <div className="card mb-8">
                <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
                  <AlertTriangle className="text-red-500" size={28} />
                  Anomaly Detection Results
                </h2>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                  <div className="bg-red-50 p-4 rounded-lg">
                    <p className="text-sm text-gray-600">Anomalies Detected</p>
                    <p className="text-3xl font-bold text-red-600">
                      {results.anomaly_detection.num_anomalies}
                    </p>
                    <p className="text-sm text-gray-500">
                      {(results.anomaly_detection.metadata.anomaly_rate * 100).toFixed(2)}% of total
                    </p>
                  </div>

                  <div className="bg-blue-50 p-4 rounded-lg">
                    <p className="text-sm text-gray-600">Detection Threshold</p>
                    <p className="text-3xl font-bold text-blue-600">
                      {results.anomaly_detection.metadata.threshold.toFixed(4)}
                    </p>
                    <p className="text-sm text-gray-500">
                      Score threshold
                    </p>
                  </div>

                  <div className="bg-green-50 p-4 rounded-lg">
                    <p className="text-sm text-gray-600">Avg Anomaly Score</p>
                    <p className="text-3xl font-bold text-green-600">
                      {results.anomaly_detection.metadata.mean_score.toFixed(4)}
                    </p>
                    <p className="text-sm text-gray-500">
                      σ = {results.anomaly_detection.metadata.std_score.toFixed(4)}
                    </p>
                  </div>
                </div>

                {/* Top Anomalies Table */}
                <div className="overflow-x-auto">
                  <h3 className="text-lg font-semibold mb-3">Top Anomalies</h3>
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Timestamp
                        </th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          IP Address
                        </th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          URL
                        </th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Status
                        </th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Score
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {results.anomaly_detection.top_anomalies.slice(0, 10).map((anomaly, idx) => (
                        <tr key={idx} className="hover:bg-gray-50">
                          <td className="px-4 py-3 text-sm text-gray-900">
                            {anomaly.timestamp ? new Date(anomaly.timestamp).toLocaleString() : 'N/A'}
                          </td>
                          <td className="px-4 py-3 text-sm text-gray-900 font-mono">
                            {anomaly.ip || 'N/A'}
                          </td>
                          <td className="px-4 py-3 text-sm text-gray-600 truncate max-w-xs">
                            {anomaly.url || 'N/A'}
                          </td>
                          <td className="px-4 py-3 text-sm">
                            <span className={`px-2 py-1 rounded ${
                              anomaly.status >= 500 ? 'bg-red-100 text-red-800' :
                              anomaly.status >= 400 ? 'bg-yellow-100 text-yellow-800' :
                              'bg-green-100 text-green-800'
                            }`}>
                              {anomaly.status || 'N/A'}
                            </span>
                          </td>
                          <td className="px-4 py-3 text-sm font-bold text-red-600">
                            {anomaly.anomaly_score.toFixed(4)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Root Cause Analysis */}
            {results.root_cause && (
              <div className="card mb-8">
                <h2 className="text-2xl font-bold mb-4">Root Cause Analysis</h2>
                <p className="text-gray-700 mb-4">{results.root_cause.summary}</p>

                <h3 className="text-lg font-semibold mb-3">Top Contributing Features</h3>
                <div className="space-y-2">
                  {results.root_cause.top_contributing_features.slice(0, 5).map((feature, idx) => (
                    <div key={idx} className="flex items-center gap-3">
                      <div className="w-32 text-sm font-medium text-gray-700">
                        {feature.feature}
                      </div>
                      <div className="flex-1 bg-gray-200 rounded-full h-4">
                        <div
                          className="bg-blue-600 h-4 rounded-full"
                          style={{ width: `${feature.percentage}%` }}
                        ></div>
                      </div>
                      <div className="w-20 text-sm text-gray-600 text-right">
                        {feature.percentage.toFixed(1)}%
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Forecasting Results */}
            {results.forecasting && (
              <div className="card mb-8">
                <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
                  <TrendingUp className="text-blue-500" size={28} />
                  Traffic Forecast
                </h2>

                {results.forecasting.forecasts && (
                  <>
                    {Object.entries(results.forecasting.forecasts).map(([horizon, forecast]) => (
                      <div key={horizon} className="mb-6">
                        <h3 className="text-lg font-semibold mb-3">
                          {horizon} Forecast
                        </h3>

                        {forecast.forecast && forecast.forecast.length > 0 && (
                          <Plot
                            data={[
                              {
                                x: forecast.forecast.map(f => f.timestamp),
                                y: forecast.forecast.map(f => f.predicted_requests),
                                type: 'scatter',
                                mode: 'lines+markers',
                                name: 'Predicted',
                                line: { color: '#3B82F6', width: 2 },
                              },
                              {
                                x: forecast.forecast.map(f => f.timestamp),
                                y: forecast.forecast.map(f => f.upper_bound),
                                type: 'scatter',
                                mode: 'lines',
                                name: 'Upper Bound',
                                line: { color: '#93C5FD', width: 1, dash: 'dash' },
                                fill: 'tonexty',
                                fillcolor: 'rgba(59, 130, 246, 0.1)',
                              },
                              {
                                x: forecast.forecast.map(f => f.timestamp),
                                y: forecast.forecast.map(f => f.lower_bound),
                                type: 'scatter',
                                mode: 'lines',
                                name: 'Lower Bound',
                                line: { color: '#93C5FD', width: 1, dash: 'dash' },
                              },
                            ]}
                            layout={{
                              title: `${horizon} Traffic Forecast`,
                              xaxis: { title: 'Time' },
                              yaxis: { title: 'Requests per Hour' },
                              height: 400,
                              showlegend: true,
                            }}
                            config={{ responsive: true }}
                            className="w-full"
                          />
                        )}
                      </div>
                    ))}
                  </>
                )}
              </div>
            )}
          </>
        )}
      </main>
    </div>
  )
}

function StatCard({ icon, title, value, subtitle }) {
  return (
    <div className="card">
      <div className="flex items-center gap-3 mb-2">
        <div className="text-blue-600">{icon}</div>
        <h3 className="text-sm font-medium text-gray-600">{title}</h3>
      </div>
      <p className="text-2xl font-bold text-gray-900">{value}</p>
      <p className="text-sm text-gray-500 mt-1">{subtitle}</p>
    </div>
  )
}

export default App
