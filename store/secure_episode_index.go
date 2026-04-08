package store

import (
	"context"
	"fmt"
)

// SecureEpisodeIndexRow stores one keyed episode representation for personal search/clustering.
type SecureEpisodeIndexRow struct {
	EpisodeUUID  string
	GroupID      string
	Method       string
	Content      string
	Source       string
	Public       []float32
	BaseWaveReal []float32
	BaseWaveImag []float32
	WaveReal     []float32
	WaveImag     []float32
	ModeWeight   []float32
	ModeEnergy   []float32
}

// ReplaceSecureEpisodeIndex replaces the keyed episode index for one group/method.
func (d *DB) ReplaceSecureEpisodeIndex(ctx context.Context, groupID, method string, rows []SecureEpisodeIndexRow) error {
	tx, err := d.sql.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback() //nolint:errcheck

	if _, err := tx.ExecContext(ctx,
		`DELETE FROM secure_episode_index WHERE group_id = ? AND method = ?`,
		groupID, method,
	); err != nil {
		return fmt.Errorf("clear secure index: %w", err)
	}

	stmt, err := tx.PrepareContext(ctx, `
		INSERT INTO secure_episode_index (
			episode_uuid, group_id, method, public_layer, base_vec, base_wave_imag, wave_real, wave_imag, mode_weight, mode_energy, key_probe
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	`)
	if err != nil {
		return err
	}
	defer stmt.Close() //nolint:errcheck

	for _, row := range rows {
		_, err := stmt.ExecContext(ctx,
			row.EpisodeUUID,
			groupID,
			method,
			EncodeEmbedding(row.Public),
			EncodeEmbedding(row.BaseWaveReal),
			EncodeEmbedding(row.BaseWaveImag),
			EncodeEmbedding(row.WaveReal),
			EncodeEmbedding(row.WaveImag),
			EncodeEmbedding(row.ModeWeight),
			EncodeEmbedding(row.ModeEnergy),
			EncodeEmbedding(nil),
		)
		if err != nil {
			return fmt.Errorf("insert secure index row %s: %w", row.EpisodeUUID, err)
		}
	}

	return tx.Commit()
}

// AllSecureEpisodes loads all keyed episode rows for one group/method, joined with the episode content.
func (d *DB) AllSecureEpisodes(ctx context.Context, groupID, method string) ([]SecureEpisodeIndexRow, error) {
	rows, err := d.sql.QueryContext(ctx, `
		SELECT
			s.episode_uuid,
			e.content,
			e.source,
			s.public_layer,
			s.base_vec,
			s.base_wave_imag,
			s.wave_real,
			s.wave_imag,
			s.mode_weight,
			s.mode_energy
		FROM secure_episode_index s
		JOIN episodes e ON e.uuid = s.episode_uuid
		WHERE s.group_id = ? AND s.method = ?
		ORDER BY e.created_at DESC, s.episode_uuid ASC
	`, groupID, method)
	if err != nil {
		return nil, err
	}
	defer rows.Close() //nolint:errcheck

	var out []SecureEpisodeIndexRow
	for rows.Next() {
		var row SecureEpisodeIndexRow
		var publicBlob []byte
		var baseRealBlob []byte
		var baseImagBlob []byte
		var waveRealBlob []byte
		var waveImagBlob []byte
		var modeWeightBlob []byte
		var modeEnergyBlob []byte
		if err := rows.Scan(
			&row.EpisodeUUID,
			&row.Content,
			&row.Source,
			&publicBlob,
			&baseRealBlob,
			&baseImagBlob,
			&waveRealBlob,
			&waveImagBlob,
			&modeWeightBlob,
			&modeEnergyBlob,
		); err != nil {
			return nil, err
		}
		row.GroupID = groupID
		row.Method = method
		row.Public = DecodeEmbedding(publicBlob)
		row.BaseWaveReal = DecodeEmbedding(baseRealBlob)
		row.BaseWaveImag = DecodeEmbedding(baseImagBlob)
		row.WaveReal = DecodeEmbedding(waveRealBlob)
		row.WaveImag = DecodeEmbedding(waveImagBlob)
		row.ModeWeight = DecodeEmbedding(modeWeightBlob)
		row.ModeEnergy = DecodeEmbedding(modeEnergyBlob)
		out = append(out, row)
	}
	return out, rows.Err()
}

// CountSecureEpisodes returns the number of keyed episode rows for one group/method.
func (d *DB) CountSecureEpisodes(ctx context.Context, groupID, method string) (int, error) {
	var n int
	err := d.sql.QueryRowContext(ctx,
		`SELECT count(*) FROM secure_episode_index WHERE group_id = ? AND method = ?`,
		groupID, method,
	).Scan(&n)
	return n, err
}
