var api = require("../../utils/api.js");

Page({
  data: {
    rows: [],
    boardModel: "",
    loading: true,
  },

  onShow: function () {
    this.startWatch();
    this.loadBoard();
  },

  onHide: function () { this.stopWatch(); },
  onUnload: function () { this.stopWatch(); },

  loadBoard: function () {
    var self = this;
    var db = wx.cloud.database();
    db.collection("board").doc("board_current").get({
      success: function (res) {
        self.setData({ boardModel: (res.data && res.data.model) || "" });
      },
      fail: function () {},
    });
  },

  startWatch: function () {
    if (this._watcher || this._timer) return;
    var self = this;
    var db = wx.cloud.database();
    var render = function (docs) {
      var rows = (docs || []).map(function (t) {
        var info = api.statusInfo(t, self.data.boardModel);
        return {
          _id: t._id,
          type: t.type,
          typeName: (api.MODELS[t.type] && api.MODELS[t.type].name) || t.type,
          title: t.title || "",
          status: t.status,
          statusText: info.text,
          statusCls: info.cls,
          time: api.fmtTime(t.created_at),
        };
      });
      self.setData({ rows: rows, loading: false });
    };
    var pull = function () {
      db.collection("tasks").orderBy("created_at", "desc").limit(30).get({
        success: function (res) { render(res.data); },
        fail: function () { self.setData({ loading: false }); },
      });
    };
    pull();
    try {
      this._watcher = db.collection("tasks")
        .orderBy("created_at", "desc").limit(30)
        .watch({
          onChange: function (snap) {
            if (snap.docs) render(snap.docs);
          },
          onError: function () {
            self.stopWatch();
            self._timer = setInterval(pull, 4000);
          },
        });
    } catch (e) {
      this._timer = setInterval(pull, 4000);
    }
  },

  stopWatch: function () {
    if (this._watcher) { this._watcher.close(); this._watcher = null; }
    if (this._timer) { clearInterval(this._timer); this._timer = null; }
  },

  openTask: function (e) {
    wx.navigateTo({ url: "/pages/task/task?id=" + e.currentTarget.dataset.id });
  },

  goHome: function () {
    wx.switchTab({ url: "/pages/home/home" });
  },
});
