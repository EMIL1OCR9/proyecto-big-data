{page === 'pacientes' && (
          <section className="patients-page">
            <div className="patients-header">
              <div className="patients-header-text">
                <span className="eyebrow">EXPLORADOR DE DATOS</span>
                <h1>
                  <span>Pacientes</span>
                </h1>
                <p>Registros de la cohorte de investigación, consultados desde tu API FastAPI.</p>
              </div>
              <button
                type="button"
                className="refresh-pill-btn"
                onClick={() => setRefresh((v) => v + 1)}
              >
                <span aria-hidden="true">↻</span> Actualizar
              </button>
            </div>

            <div className="filters-panel">
              <label>
                <span>División de datos</span>
                <select
                  value={split}
                  onChange={(event) => {
                    setSplit(event.target.value as Split)
                    setOffset(0)
                    setSearch('')
                  }}
                >
                  <option value="">Todos</option>
                  <option value="train">Entrenamiento</option>
                  <option value="validation">Validación</option>
                  <option value="test">Prueba</option>
                </select>
              </label>

              <label>
                <span>Cáncer a 12 meses (NLST)</span>
                <select
                  value={cancer}
                  onChange={(event) => {
                    setCancer(event.target.value as Cancer)
                    setOffset(0)
                  }}
                >
                  <option value="">Todos</option>
                  <option value="true">Sí tuvo cáncer</option>
                  <option value="false">No tuvo cáncer</option>
                </select>
              </label>

              <label>
                <span>Nódulo mayor de al menos (mm)</span>
                <input
                  type="number"
                  min="0"
                  step="1"
                  value={minSize}
                  onChange={(event) => {
                    setMinSize(event.target.value)
                    setOffset(0)
                  }}
                  placeholder="Por ejemplo, 8"
                />
              </label>

              <label>
                <span>Ordenar por</span>
                <select
                  value={order}
                  onChange={(event) => {
                    setOrder(event.target.value as Order)
                    setOffset(0)
                  }}
                >
                  <option value="id">ID de paciente</option>
                  <option value="nodulos">Más nódulos</option>
                  <option value="tamano">Nódulo más grande</option>
                </select>
              </label>

              <label>
                <span>Filtrar ID en esta página</span>
                <input
                  value={search}
                  onChange={(event) => setSearch(event.target.value)}
                  placeholder="Buscar en los registros visibles"
                />
              </label>
            </div>

            {(cancer || minSize || order !== 'id') && (
              <p className="footnote">
                La primera consulta con filtros tarda unos segundos: lee los resultados de todos los
                pacientes.
              </p>
            )}

            {error && (
              <div className="alert">
                No pudimos leer los pacientes: {error}.{' '}
                <button onClick={() => setRefresh((v) => v + 1)}>Reintentar</button>
              </div>
            )}

            <div className="patients-table-card">
              <div className="patients-table-header">
                <strong>Listado de pacientes</strong>
                <span className="records-badge">
                  {list ? `${list.total.toLocaleString('es-MX')} registros` : 'Sin conexión'}
                </span>
              </div>

              <div className="table-scroll">
                <table className="patients-table">
                  <thead>
                    <tr>
                      <th>Paciente</th>
                      <th>Series</th>
                      <th>Análisis</th>
                      <th>Nódulos detectados</th>
                      <th>Mayor tamaño</th>
                      <th>Cáncer a 12 meses</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {loading ? (
                      <tr>
                        <td colSpan={7} className="empty-cell">
                          Cargando pacientes…
                        </td>
                      </tr>
                    ) : patients.length ? (
                      patients.map((patient) => (
                        <tr key={patient.patient_id}>
                          <td className="mono">{patient.patient_id}</td>
                          <td>{patient.series_total}</td>
                          <td>
                            <span className={`state state-${patient.estado}`}>
                              {statusLabel(patient.estado)}
                            </span>
                          </td>
                          <td>{patient.nodulos_detectados ?? '—'}</td>
                          <td>
                            {patient.nodulo_mayor_mm == null ? '—' : `${patient.nodulo_mayor_mm} mm`}
                          </td>
                          <td>
                            {patient.cancer_12m == null ? '—' : patient.cancer_12m ? 'Sí' : 'No'}
                          </td>
                          <td className="action-cell">
                            <button
                              type="button"
                              className="detail-link-btn"
                              onClick={() => setSelectedId(patient.patient_id)}
                            >
                              Ver detalle →
                            </button>
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan={7} className="empty-cell">
                          No hay pacientes para esta consulta.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>

              <div className="patients-pagination">
                <span>
                  {list
                    ? `${offset + (list.pacientes.length ? 1 : 0)}–${
                        offset + list.pacientes.length
                      } de ${list.total.toLocaleString('es-MX')}`
                    : '—'}
                </span>
                <div className="pagination-controls">
                  <button
                    type="button"
                    disabled={loading || offset === 0}
                    onClick={() => {
                      setOffset((v) => Math.max(0, v - PAGE_SIZE))
                      setSearch('')
                    }}
                  >
                    Anterior
                  </button>
                  <button
                    type="button"
                    disabled={loading || !list || offset + PAGE_SIZE >= list.total}
                    onClick={() => {
                      setOffset((v) => v + PAGE_SIZE)
                      setSearch('')
                    }}
                  >
                    Siguiente
                  </button>
                </div>
              </div>
            </div>

            <p className="footnote">
              “Pendiente” significa que todavía no existe un JSON del detector, no que se hayan
              encontrado cero nódulos.
            </p>
          </section>
        )}