"""Estimate the electricity used while processing, with CodeCarbon's OFFLINE tracker
(no internet needed, Canadian grid). If CodeCarbon isn't installed, kwh stays None.

with EnergyMeter() as meter:
    ...do work...
print(meter.kwh)
"""


class EnergyMeter:
    def __init__(self):
        self.kwh = None
        self._tracker = None

    def __enter__(self):
        try:
            from codecarbon import OfflineEmissionsTracker
            self._tracker = OfflineEmissionsTracker(
                country_iso_code="CAN", save_to_file=False, log_level="error")
            self._tracker.start()
        except Exception:
            self._tracker = None
        return self

    def __exit__(self, *exc):
        if self._tracker is not None:
            try:
                self._tracker.stop()
                self.kwh = float(self._tracker.final_emissions_data.energy_consumed)
            except Exception:
                self.kwh = None
        return False


def describe_kwh(kwh):
    """Turn kWh into something people understand."""
    if kwh is None:
        return "not measured"
    wh = kwh * 1000
    phone_charges = wh / 15  # a phone battery holds roughly 15 Wh
    if phone_charges < 0.1:
        return f"{wh:.2f} Wh (less than a tenth of one phone charge)"
    return f"{wh:.2f} Wh (about {phone_charges:.1f} phone charges)"
