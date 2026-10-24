// Map coordinates are metres. All control points remain in the scene's route array.
export function mountMap(canvas, getScene, onEditEnd) {
  let map = null;
  let view = {x: 0, y: 80, scale: 9};
  let drag = null;
  let readonly = false;
  let trajectory = null;
  let frame = 0;

  function screen(point) {
    return [canvas.clientWidth / 2 + (point[0] - view.x) * view.scale,
            canvas.clientHeight / 2 - (point[1] - view.y) * view.scale];
  }
  function world(point) {
    return [view.x + (point[0] - canvas.clientWidth / 2) / view.scale,
            view.y - (point[1] - canvas.clientHeight / 2) / view.scale];
  }
  function cameraArrow(ctx, x, y, dx, dy, color, label) {
    const tipX = x + dx * 33, tipY = y - dy * 33;
    const sideX = -dy * 6, sideY = -dx * 6;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(tipX, tipY); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(tipX, tipY);
    ctx.lineTo(tipX - dx * 10 + sideX, tipY + dy * 10 + sideY);
    ctx.lineTo(tipX - dx * 10 - sideX, tipY + dy * 10 - sideY);
    ctx.closePath(); ctx.fill();
    ctx.font = 'bold 12px sans-serif';
    ctx.fillText(label, tipX + 5, tipY - 4);
  }
  function draw() {
    const scene = getScene();
    if (!scene || !map) return;
    const ratio = devicePixelRatio || 1;
    canvas.width = Math.max(1, canvas.clientWidth * ratio);
    canvas.height = Math.max(1, canvas.clientHeight * ratio);
    const ctx = canvas.getContext('2d');
    ctx.scale(ratio, ratio);
    ctx.fillStyle = '#69766e';
    ctx.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);
    for (const road of map.roads) {
      const a = screen([road[0], road[3]]), b = screen([road[2], road[1]]);
      ctx.fillStyle = '#548b63';
      ctx.fillRect(a[0], a[1], b[0] - a[0], b[1] - a[1]);
    }
    for (const person of scene.people) {
      ctx.strokeStyle = '#ffda83';
      ctx.lineWidth = 3;
      ctx.beginPath();
      person.route_xy_m.forEach((point, index) => {
        const p = screen(point);
        if (!index) ctx.moveTo(...p); else ctx.lineTo(...p);
      });
      ctx.stroke();
      person.route_xy_m.forEach((point, index) => {
        const p = screen(point);
        ctx.fillStyle = '#fff1bb';
        ctx.beginPath(); ctx.arc(...p, 7, 0, 2 * Math.PI); ctx.fill();
        ctx.fillStyle = '#17252c';
        ctx.fillText(`P${person.person_id}·${index + 1}`, p[0] + 9, p[1] - 7);
      });
    }
    for (const person of scene.people) {
      const position = trajectory?.people?.[String(person.person_id)]?.[frame];
      if (!position) continue;
      const [x, y] = screen(position.xy);
      const [dx, dy] = position.direction;
      ctx.fillStyle = '#f86b35';
      ctx.strokeStyle = '#17252c';
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + dx * 19, y - dy * 19); ctx.stroke();
      ctx.beginPath(); ctx.arc(x, y, 11, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      ctx.fillStyle = '#17252c';
      ctx.font = 'bold 11px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(String(person.person_id), x, y);
      ctx.textAlign = 'start';
      ctx.textBaseline = 'alphabetic';
    }
    for (const rig of scene.rigs) {
      const p = screen(rig.origin_m);
      // camera/rig.py: cam2's physical optical axis is yaw * (0, +1);
      // cam4 is opposite. The 0.18 m camera spacing is too small to draw to scale.
      const yaw = rig.yaw_deg * Math.PI / 180;
      const dx = -Math.sin(yaw), dy = Math.cos(yaw);
      cameraArrow(ctx, p[0], p[1], dx, dy, '#80e5ed', 'cam2');
      cameraArrow(ctx, p[0], p[1], -dx, -dy, '#d29cfa', 'cam4');
      ctx.fillStyle = '#80e5ed';
      ctx.beginPath(); ctx.arc(...p, 8, 0, 2 * Math.PI); ctx.fill();
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 12px sans-serif';
      ctx.fillText(`${rig.rig_id} · ${Number(rig.yaw_deg).toFixed(1)}°`, p[0] + 10, p[1] + 20);
    }
  }
  function pointer(event) {
    const rect = canvas.getBoundingClientRect();
    return [event.clientX - rect.left, event.clientY - rect.top];
  }
  canvas.addEventListener('pointerdown', event => {
    const p = pointer(event), candidates = [];
    if (!readonly) {
      for (const person of getScene().people) {
        person.route_xy_m.forEach((point, index) =>
          candidates.push({kind: 'point', person, index, distance: Math.hypot(...screen(point).map((v, i) => v - p[i]))}));
      }
      for (const rig of getScene().rigs) {
        candidates.push({kind: 'rig', rig, distance: Math.hypot(...screen(rig.origin_m).map((v, i) => v - p[i]))});
      }
    }
    candidates.sort((a, b) => a.distance - b.distance);
    drag = candidates[0]?.distance < 17 ? candidates[0] : {kind: 'pan', point: p, view: {...view}};
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener('pointermove', event => {
    if (!drag) return;
    const p = pointer(event);
    if (drag.kind === 'point') {
      drag.person.route_xy_m[drag.index] = world(p);
      trajectory = null;
    }
    else if (drag.kind === 'rig') {
      drag.rig.origin_m[0] = world(p)[0];
      drag.rig.origin_m[1] = world(p)[1];
    } else {
      view.x = drag.view.x - (p[0] - drag.point[0]) / view.scale;
      view.y = drag.view.y + (p[1] - drag.point[1]) / view.scale;
    }
    draw();
  });
  canvas.addEventListener('pointerup', () => {
    if (drag && drag.kind !== 'pan') onEditEnd();
    drag = null;
  });
  canvas.addEventListener('wheel', event => {
    event.preventDefault();
    view.scale = Math.max(2, Math.min(40, view.scale * (event.deltaY < 0 ? 1.15 : 0.87)));
    draw();
  }, {passive: false});
  new ResizeObserver(draw).observe(canvas);
  return {
    draw,
    setFrame(value) { frame = value; draw(); },
    setTrajectory(value) { trajectory = value; draw(); },
    setMap(value, environment) {
      map = value;
      view = environment === 'city' ? {x: 0, y: 81, scale: 9} : {x: 0, y: 2, scale: 14};
      draw();
    },
    setReadonly(value) { readonly = value; draw(); },
  };
}
